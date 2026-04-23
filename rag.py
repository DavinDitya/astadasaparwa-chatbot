# rag.py
import os
import json
import re
import numpy as np
import faiss
from google import genai
from google.genai import types
from gemini_client import generate_with_gemini
from dotenv import load_dotenv

load_dotenv()

# === Konfigurasi ===
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
EMBED_MODEL_NAME = "gemini-embedding-001"

_index = None
_chunks = None
_vectors = None
_book_names = None

DEFAULT_TOP_K = 10 
CONTEXT_CHAR_LIMIT = 8000 

# -------------------------
# 1. Utilities & Cleaning
# -------------------------
def clean_markdown(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'\*', '', text)     
    
    patterns = [
        r"^berdasarkan (kutipan )?teks( yang diberikan)?( ini)?(\.|,)?",
        r"^mengacu pada (kutipan )?teks(\.|,)?",
        r"^menurut (kutipan )?teks(\.|,)?",
        r"^dalam (kutipan )?teks( ini)?(\.|,)?",
        r"^teks (ini )?menjelaskan( bahwa)?",
        r"^teks (ini )?tidak (menjelaskan|menyebutkan)",
        r"^adp ai menjawab:",
        r"^kutipan (teks )?tersebut menceritakan",
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    if text:
        text = text[0].upper() + text[1:]

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_book_name(s: str) -> str:
    if not s: return ""
    return " ".join(s.lower().replace("w", "v").split())

# (FUNGSI REWRITE_QUERY DIHAPUS SEPENUHNYA UNTUK OPTIMASI)

# -------------------------
# 2. Search Logic
# -------------------------
def load_store():
    global _index, _chunks, _vectors, _book_names
    if _index is None:
        if not os.path.exists(INDEX_PATH): raise FileNotFoundError("Index not found")
        _index = faiss.read_index(INDEX_PATH)
    if _chunks is None:
        with open(CHUNKS_JSON, "r", encoding="utf-8") as f: _chunks = json.load(f)
    if _vectors is None and os.path.exists(VECTORS_NPY):
        _vectors = np.load(VECTORS_NPY)
        norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _vectors = _vectors / norms
    
    if _book_names is None:
        names = {normalize_book_name(c.get("meta", {}).get("book")) for c in _chunks if c.get("meta", {}).get("book")}
        _book_names = sorted(list(names))

def detect_book_from_query(q: str) -> list:
    load_store()
    q_norm = normalize_book_name(q)
    return [b for b in _book_names if b in q_norm] if _book_names else []

def retrieve_candidates(query: str, top_k: int = DEFAULT_TOP_K, book_filter: list = None):
    idx, chunks, vectors = _index, _chunks, _vectors
    
    response = client.models.embed_content(
        model=EMBED_MODEL_NAME,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    
    q_vec = np.array([response.embeddings[0].values], dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    if book_filter and vectors is not None:
        indices = [i for i, c in enumerate(chunks) if normalize_book_name(c.get("meta", {}).get("book")) in book_filter]
        if indices:
            scores = np.dot(vectors[indices], q_vec[0])
            top_ids = np.argsort(-scores)[:top_k*4] 
            return [(float(scores[i]), indices[i]) for i in top_ids]
    
    D, I = idx.search(q_vec, top_k * 4)
    return [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]

def build_context_text(retrieved: list) -> str:
    texts = []
    total_len = 0
    for r in retrieved:
        body = r["chunk"].get("text_en", "") 
        if total_len + len(body) > CONTEXT_CHAR_LIMIT: break
        texts.append(body)
        total_len += len(body)
    return "\n---\n".join(texts)

# -------------------------
# 3. Prompting (Ditambah Logika Mode)
# -------------------------
def build_prompt(question: str, retrieved_chunks: list, mode: str) -> str:
    context_str = build_context_text(retrieved_chunks)
    
    # [BARU] Logika instruksi berdasarkan pilihan user
    if mode == "singkat":
        instruksi_panjang = "Jawablah dengan SINGKAT, PADAT, dan JELAS (maksimal 2-3 kalimat saja) langsung pada intinya."
    else:
        instruksi_panjang = "Jawablah dengan SANGAT DETAIL dan PANJANG. Ceritakan latar belakang, tokoh yang terlibat, dan akhir ceritanya secara runut."

    prompt = f"""
    Anda adalah 'ADP AI' (Asisten Asta Dasa Parwa).
    
    ATURAN JAWAB:
    1. JAWAB LANGSUNG ke inti pertanyaan.
    2. DILARANG KERAS menggunakan kata: "Berdasarkan teks", "Teks menyebutkan", atau "Menurut kutipan".
    3. Jika informasi tidak ada di teks, gunakan pengetahuan umum tentang Mahabharata.
    4. Gunakan Bahasa Indonesia yang natural dan bercerita.
    5. {instruksi_panjang}
    
    DATA REFERENSI:
    {context_str}

    PERTANYAAN: {question}

    JAWABAN:
    """
    return prompt.strip()

# -------------------------
# 4. Main Pipeline
# -------------------------
def answer_question(question: str, history: list = [], top_k: int = DEFAULT_TOP_K, mode: str = "detail"):
    try:
        load_store()
        
        # 1. Langsung pakai pertanyaan asli (Lewati Rewriter yang bikin lambat!)
        search_q = question
        if any(x in search_q.lower() for x in ["siapa", "tokoh", "sifat"]):
            search_q += " deskripsi karakter peran asal usul"
            
        # 2. Retrieval ke FAISS
        books = detect_book_from_query(search_q)
        candidates = retrieve_candidates(search_q, top_k, books)
        
        retrieved_data = []
        for score, idx in candidates:
            retrieved_data.append({"score": score, "chunk": _chunks[idx]})
        
        best_chunks = retrieved_data[:top_k]

        # 3. Generate Answer dengan Mode (Singkat/Detail)
        prompt = build_prompt(question, best_chunks, mode)
        raw_response = generate_with_gemini(prompt)

        # 4. Fallback jika gagal (503 Error)
        if raw_response is None:
            print("⚠️ Fallback Mode Aktif: Gemini Error 503")
            # Coba pancing sekali lagi dengan prompt super simpel, siapa tau servernya sudah lowong
            fallback_prompt = f"Ceritakan secara singkat tentang {question} di Mahabharata."
            raw_response = generate_with_gemini(fallback_prompt, max_tokens=150)

        final_answer = clean_markdown(raw_response) if raw_response else "Maaf, server sedang sibuk (Error 503). Mohon tunggu beberapa detik lalu coba lagi."
        
        return {"answer": final_answer, "retrieved": best_chunks}

    except Exception as e:
        print(f"[ERROR RAG] {e}")
        return {"answer": "Maaf, sistem sedang sibuk.", "retrieved": []}