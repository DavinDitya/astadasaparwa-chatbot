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

# Konfigurasi Gemini API Baru
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
    
    # 1. Hapus Markdown Bold/Italic
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'\*', '', text)     
    
    # 2. Hapus Prefix "Robot" (AGRESIF)
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

    # 3. Capitalize huruf pertama setelah dipotong
    if text:
        text = text[0].upper() + text[1:]

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_book_name(s: str) -> str:
    if not s: return ""
    return " ".join(s.lower().replace("w", "v").split())

# -------------------------
# 2. CONTEXTUAL REWRITER
# -------------------------
def rewrite_query(question: str, history: list) -> str:
    if not history:
        return question
    
    last_turn = history[-2:] 
    history_str = ""
    for h in last_turn:
        role = "User" if h.get('role') == 'user' else "Assistant"
        content = h.get('content', '')
        history_str += f"{role}: {content}\n"

    # PROMPT BARU: Jauh lebih galak dan tegas!
    prompt = f"""
    Tugasmu HANYA menulis ulang pertanyaan terakhir User agar menjadi kalimat lengkap dan berdiri sendiri (Standalone Question) berdasarkan konteks percakapan.
    
    ATURAN MUTLAK:
    1. Ganti kata ganti (dia, -nya, mereka, itu) dengan nama tokoh/subjek yang sedang dibicarakan sebelumnya.
    2. DILARANG menjawab pertanyaannya.
    3. DILARANG menggunakan kata pengantar (seperti: "Siap", "Baik", "Tentu", "Ini pertanyaannya").
    4. Jika pertanyaan sudah jelas tanpa perlu diubah, kembalikan saja pertanyaan aslinya secara utuh.
    
    Percakapan Sebelumnya:
    {history_str}
    
    Pertanyaan User Saat Ini: {question}
    
    Standalone Question:
    """
    
    rewritten = generate_with_gemini(prompt, max_tokens=100)
    if rewritten:
        # Bersihkan jika AI masih bandel menambahkan label
        cleaned = rewritten.replace("Standalone Question:", "").replace("**", "").replace('"', '').strip()
        
        # Mencegah AI menjawab "Siap" atau "Baik"
        if len(cleaned) < 5 or cleaned.lower() in ["siap", "baik", "oke", "ya", "mengerti"]:
            print(f"⚠️ Rewriter ngaco ('{cleaned}'). Pakai query asli.")
            return question
            
        print(f"🔄 Rewritten Query: '{question}' -> '{cleaned}'")
        return cleaned

    return question

# -------------------------
# 3. Search Logic
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
    
    # 1. Konversi pertanyaan menjadi Vektor menggunakan Gemini
    response = client.models.embed_content(
        model=EMBED_MODEL_NAME,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    
    # 2. Ambil nilai angka dan normalisasi
    q_vec = np.array([response.embeddings[0].values], dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    # 3. Proses pencarian kesamaan (Dot Product)
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
# 4. Prompting
# -------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_str = build_context_text(retrieved_chunks)
    
    prompt = f"""
    Anda adalah 'ADP AI' (Asisten Asta Dasa Parwa).
    
    ATURAN JAWAB:
    1. JAWAB LANGSUNG ke inti pertanyaan.
    2. DILARANG KERAS menggunakan kata: "Berdasarkan teks", "Teks menyebutkan", "Menurut kutipan", atau "Teks tidak menjelaskan".
    3. Jika informasi tidak ditemukan secara eksplisit, gunakan logika cerdas untuk menyimpulkan dari konteks yang ada (INFERENSI), atau jawab berdasarkan pengetahuan umum tentang Mahabharata jika sangat mendasar (misal: Sengkuni tokoh licik, Kurawa ada 100).
    4. Gunakan Bahasa Indonesia yang natural dan bercerita.
    
    DATA REFERENSI:
    {context_str}

    PERTANYAAN: {question}

    JAWABAN:
    """
    return prompt.strip()

# -------------------------
# 5. Main Pipeline
# -------------------------
def answer_question(question: str, history: list = [], top_k: int = DEFAULT_TOP_K):
    try:
        load_store()
        
        # 1. Rewrite Query berdasarkan History
        standalone_q = rewrite_query(question, history)
        
        # 2. Retrieval pakai Query yang sudah diperjelas
        search_q = standalone_q
        if any(x in search_q.lower() for x in ["siapa", "tokoh", "sifat"]):
            search_q += " deskripsi karakter peran asal usul"
            
        books = detect_book_from_query(search_q)
        candidates = retrieve_candidates(search_q, top_k, books)
        
        retrieved_data = []
        for score, idx in candidates:
            retrieved_data.append({"score": score, "chunk": _chunks[idx]})
        
        best_chunks = retrieved_data[:top_k]

        # 3. Generate Answer
        prompt = build_prompt(standalone_q, best_chunks)
        raw_response = generate_with_gemini(prompt)

        # 4. Fallback jika blocked
        if raw_response is None:
            print("⚠️ Fallback Mode Aktif")
            fallback_prompt = f"Ceritakan secara singkat dan halus tentang: {standalone_q}. Hindari kekerasan eksplisit."
            raw_response = generate_with_gemini(fallback_prompt)

        final_answer = clean_markdown(raw_response) if raw_response else "Maaf, terjadi gangguan."
        
        return {"answer": final_answer, "retrieved": best_chunks}

    except Exception as e:
        print(f"[ERROR RAG] {e}")
        return {"answer": "Maaf, sistem sedang sibuk.", "retrieved": []}