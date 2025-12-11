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

# Cache Global
_index = None
_chunks = None
_vectors = None
_book_names = None

# Configs
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", 4))
CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", 3500))
METADATA_BOOST_FACTOR = 1.15

# -------------------------
# 1. Utilities & Cleaning
# -------------------------
def clean_markdown(text: str) -> str:
    """Membersihkan simbol Markdown agar rapi di Android."""
    if not text: return ""
    # Hapus bold/italic marker
    text = re.sub(r'\*\*|__', '', text) 
    text = re.sub(r'\*', '', text)
    # Kurangi enter berlebih
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_book_name(s: str) -> str:
    if not s: return ""
    return " ".join(s.lower().replace("w", "v").split())

def expand_question(question: str) -> str:
    q = question.lower().strip()
    # Expansion sederhana
    if "isi" in q or "tentang" in q:
        return question + ". ceritakan ringkasan dan tokoh utama"
    return question

# -------------------------
# 2. Load Store & Search Logic
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
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Search Logic
    if book_filter and vectors is not None:
        # Filtered search (Fast)
        indices = [i for i, c in enumerate(chunks) if normalize_book_name(c.get("meta", {}).get("book")) in book_filter]
        if indices:
            scores = np.dot(vectors[indices], q_vec)
            top_ids = np.argsort(-scores)[:top_k*3]
            return [(float(scores[i]), indices[i]) for i in top_ids]
    
    # Normal FAISS Search
    D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 5)
    return [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]

def rerank(query: str, candidates: list, book_mentions: list = None) -> list:
    _, chunks, precomputed = _index, _chunks, _vectors
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    scored = []
    for faiss_score, idx in candidates:
        score = float(np.dot(q_vec, precomputed[idx])) if precomputed is not None else float(faiss_score)
        
        # Boost score jika buku sesuai
        book = normalize_book_name(chunks[idx].get("meta", {}).get("book", ""))
        if book_mentions and book in book_mentions:
            score *= METADATA_BOOST_FACTOR
        scored.append((score, idx))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate & Format
    results = []
    seen = set()
    for score, idx in scored:
        if idx not in seen:
            seen.add(idx)
            results.append({"score": score, "chunk": _chunks[idx]})
    return results

def build_context_text(retrieved: list) -> str:
    texts = []
    total_len = 0
    for r in retrieved:
        body = r["chunk"].get("text_en", "")
        if total_len + len(body) > CONTEXT_CHAR_LIMIT: break
        texts.append(body)
        total_len += len(body)
    return "\n\n".join(texts)

# -------------------------
# 3. Prompting (VERSI NETRAL & AMAN)
# -------------------------
def build_prompt(question: str, retrieved_chunks: list) -> str:
    context_str = build_context_text(retrieved_chunks)
    
    prompt = f"""
    Task: Rewrite and Translate.
    
    Below is an excerpt from the ancient Mahabharata epic (public domain literature).
    Your goal is to summarize the excerpt into Indonesian language to answer the user's query.
    
    User Query: {question}
    
    Source Text (Literature):
    {context_str}
    
    Instructions:
    1. Treat this purely as a literary summarization task.
    2. Summarize the events in the Source Text that answer the query.
    3. Output ONLY the Indonesian summary.
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
        retrieved = rerank(expanded_q, candidates, books)
        
        # Ambil 1 chunk terbaik saja agar aman dari Safety Filter
        limited_retrieved = retrieved[:1] 

        # 2. Generate (Percobaan 1)
        prompt = build_prompt(question, limited_retrieved)
        response_text = generate_with_gemini(prompt)

        # 3. Cek Blokir & Fallback
        block_indicators = ["diblokir", "safety", "finish_reason", "tidak ada teks"]
        
        # Cek response_text apakah valid
        is_blocked = False
        if not response_text or any(x in str(response_text).lower() for x in block_indicators):
            is_blocked = True
            
        if is_blocked:
            print(f"⚠️ Terblokir. Mencoba Fallback AMAN untuk: {question}")
            
            # FALLBACK STRATEGI BARU:
            # Jangan gunakan Context RAG sama sekali (karena Context-nya yang bikin blokir).
            # Minta Gemini menggunakan pengetahuan umumnya (internal knowledge) tapi sangat singkat.
            
            fallback_prompt = f"""
            Jelaskan secara singkat dan sopan ringkasan dari bagian Mahabharata ini: "{question}".
            Fokus pada tema utama dan tokoh-tokohnya saja.
            JANGAN menyebutkan detail kekerasan, pembunuhan, atau darah.
            Gunakan bahasa eufemisme (penghalus) untuk konflik.
            """
            
            response_text = generate_with_gemini(fallback_prompt)

        # 4. Cleaning Akhir
        final_answer = clean_markdown(response_text)
        
        return {"answer": final_answer, "retrieved": retrieved[:top_k]}

    except Exception as e:
        print(f"[ERROR] {e}")
        return {"answer": "Maaf, terjadi kesalahan sistem.", "retrieved": []}

if __name__ == "__main__":
    # Test area
    q = "Apa isi utama dari Adi Parva?"
    print(f"Testing: {q}")
    res = answer_question(q)
    print("Jawaban:", res['answer'])