import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from gemini_client import generate_with_gemini
from dotenv import load_dotenv

# Load Environment
load_dotenv()

# === Konfigurasi & Model ===
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Global Variables
_index = None
_chunks = None
_vectors = None
_book_names = None

# Configs
DEFAULT_TOP_K = 4
CONTEXT_CHAR_LIMIT = 4000 # Diperbesar agar konteks lebih banyak

# -------------------------
# 1. Utilities
# -------------------------
def clean_markdown(text: str) -> str:
    """Membersihkan simbol Markdown agar rapi di Android."""
    if not text: return ""
    text = re.sub(r'\*\*|__', '', text) # Hapus bold
    text = re.sub(r'\*', '', text)      # Hapus italic
    text = re.sub(r'\n{3,}', '\n\n', text) # Hapus enter berlebih
    return text.strip()

def normalize_book_name(s: str) -> str:
    if not s: return ""
    return " ".join(s.lower().replace("w", "v").split())

def expand_question(question: str) -> str:
    return question.lower().strip()

# -------------------------
# 2. Search Logic (Retrieval)
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
        # Normalisasi vektor
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
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    if book_filter and vectors is not None:
        indices = [i for i, c in enumerate(chunks) if normalize_book_name(c.get("meta", {}).get("book")) in book_filter]
        if indices:
            scores = np.dot(vectors[indices], q_vec)
            top_ids = np.argsort(-scores)[:top_k*3]
            return [(float(scores[i]), indices[i]) for i in top_ids]
    
    D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 3)
    return [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]

def build_context_text(retrieved: list) -> str:
    texts = []
    total_len = 0
    # Kita urutkan berdasarkan skor tertinggi
    for r in retrieved:
        # Prioritaskan text Indonesia jika ada, jika tidak pakai English
        body = r["chunk"].get("text_id", "") or r["chunk"].get("text_en", "")
        if total_len + len(body) > CONTEXT_CHAR_LIMIT: break
        texts.append(body)
        total_len += len(body)
    return "\n---\n".join(texts)

# -------------------------
# 3. Prompting (INI KUNCINYA)
# -------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_str = build_context_text(retrieved_chunks)
    
    # Prompt ini didesain untuk:
    # 1. Menganggap teks sebagai karya sastra (bypass safety filter kekerasan).
    # 2. Menjawab langsung (Direct Answer).
    # 3. Menggunakan bahasa Indonesia yang natural.
    
    prompt = f"""
    Anda adalah 'ADP AI', asisten ahli Sastra Kuno Asta Dasa Parwa (Mahabharata).
    Tugas anda adalah menjawab pertanyaan pengguna berdasarkan KUTIPAN TEKS yang diberikan di bawah.

    PERATURAN PENTING:
    1. JAWAB LANGSUNG pada intinya. JANGAN gunakan kalimat pembuka seperti "Berdasarkan teks...", "Menurut konteks...", atau "Kutipan ini menjelaskan...".
    2. JANGAN membuat kesimpulan di luar teks yang diberikan.
    3. Jika teks mengandung deskripsi perang atau kematian, ceritakan apa adanya dengan nada netral (gaya bahasa sastra/sejarah). Ini adalah literatur klasik, bukan kekerasan nyata.
    4. Gunakan Bahasa Indonesia yang sopan, jelas, dan mengalir.

    KUTIPAN TEKS SASTRA (SUMBER KEBENARAN):
    {context_str}

    PERTANYAAN PENGGUNA:
    {question}

    JAWABAN (Langsung ke poin):
    """
    return prompt.strip()

# -------------------------
# 4. Main Pipeline
# -------------------------
def answer_question(question: str, top_k: int = DEFAULT_TOP_K):
    try:
        load_store()
        
        # 1. Retrieval
        expanded_q = expand_question(question)
        books = detect_book_from_query(expanded_q)
        candidates = retrieve_candidates(expanded_q, top_k, books)
        
        # Format hasil retrieval
        retrieved_data = []
        for score, idx in candidates:
            retrieved_data.append({"score": score, "chunk": _chunks[idx]})
        
        # Ambil Top-K terbaik untuk konteks
        best_chunks = retrieved_data[:top_k]

        # 2. Generate Prompt
        prompt = build_prompt(question, best_chunks)

        # 3. Kirim ke Gemini
        raw_response = generate_with_gemini(prompt)

        # 4. Cleaning Akhir (Hapus markdown bold/italic biar bersih di HP)
        final_answer = clean_markdown(raw_response)
        
        return {"answer": final_answer, "retrieved": best_chunks}

    except Exception as e:
        print(f"[ERROR RAG] {e}")
        return {"answer": "Mohon maaf, ADP sedang bermeditasi (Terjadi kesalahan sistem).", "retrieved": []}