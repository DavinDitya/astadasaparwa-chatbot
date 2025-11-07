import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from gemini_client import generate_with_gemini
from dotenv import load_dotenv

# Optional: untuk deteksi bahasa query agar prompt bisa menyesuaikan
try:
    from langdetect import detect
except ImportError:
    detect = lambda x: "id"  # fallback jika langdetect tidak terpasang

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")

# Model multilingual agar retrieval Bahasa Indonesia lebih stabil
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# cache global
_index, _chunks, _vectors = None, None, None

# Konfigurasi
DEFAULT_TOP_K = 4
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.28))
CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", 3500))


# ============================================================
# LOAD DATA
# ============================================================
def load_store():
    """
    Load FAISS index, metadata JSON, dan precomputed vectors (jika ada).
    Dipanggil sekali di awal aplikasi.
    """
    global _index, _chunks, _vectors

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
        print("🔹 Loading FAISS index...")
        _index = faiss.read_index(INDEX_PATH)

    if _chunks is None:
        if not os.path.exists(CHUNKS_JSON):
            raise FileNotFoundError(f"Chunks JSON not found: {CHUNKS_JSON}")
        with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        print(f"✅ Metadata loaded. Total chunks: {len(_chunks)}")

    if _vectors is None and os.path.exists(VECTORS_NPY):
        _vectors = np.load(VECTORS_NPY)
        norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _vectors = _vectors / norms
        print("✅ Precomputed vectors loaded.")

    return _index, _chunks, _vectors


# ============================================================
# RETRIEVAL
# ============================================================
def retrieve_candidates(query: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, int]]:
    """
    Ambil kandidat hasil FAISS (skor + index).
    """
    idx, chunks, _ = load_store()
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 4)
    results = [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]
    return results


def rerank(query: str, candidates: List[Tuple[float, int]], top_k: int = DEFAULT_TOP_K) -> List[dict]:
    """
    Re-rank hasil FAISS menggunakan cosine similarity manual.
    """
    _, chunks, precomputed = load_store()
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    scored = []
    for faiss_score, idx in candidates:
        if precomputed is not None and 0 <= idx < precomputed.shape[0]:
            score = float(np.dot(q_vec, precomputed[idx]))
        else:
            score = float(faiss_score)
        scored.append((score, idx))

    # sort descending, buang duplikat dan low-score
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    results = []
    for score, idx in scored:
        if idx in seen or score < SCORE_THRESHOLD:
            continue
        seen.add(idx)
        chunk = chunks[idx]
        results.append({"score": score, "chunk": chunk})
        if len(results) >= top_k:
            break
    return results


# ============================================================
# PROMPT BUILDING
# ============================================================
def build_context_text(retrieved: List[dict], char_limit: int = CONTEXT_CHAR_LIMIT) -> str:
    """
    Gabungkan teks chunk dengan batas karakter tertentu.
    """
    parts, total = [], 0
    for r in retrieved:
        meta = r["chunk"].get("meta", {})
        header = f"[{meta.get('book')}] {meta.get('judul') or ''} (parwa_id={r['chunk'].get('parwa_id')}, chunk={r['chunk'].get('chunk_id')})"
        body = r["chunk"].get("text_en", "")
        snippet = f"{header}\n{body}"

        if total + len(snippet) > char_limit:
            snippet = snippet[: max(0, char_limit - total - 20)] + " ... (terpotong)"
            parts.append(snippet)
            break
        parts.append(snippet)
        total += len(snippet)

    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, retrieved: List[dict]) -> str:
    """
    Bangun prompt sesuai bahasa pertanyaan.
    """
    lang = "id"
    try:
        lang = detect(question)
    except Exception:
        pass

    if not retrieved:
        return f"""
Anda asisten ahli teks Asta Dasa Parwa. Pertanyaan pengguna:
{question}

Tidak ditemukan konteks relevan. Jawab ringkas:
"Maaf, saya tidak menemukan informasi yang relevan di sumber yang tersedia."
""".strip()

    context_text = build_context_text(retrieved)

    if lang == "en":
        prompt = f"""
You are an expert assistant on the Asta Dasa Parwa (Mahabharata) text.
Use **only** the context below to answer the question.
If the answer is not found in the context, say:
"I'm sorry, I couldn’t find relevant information in the available sources."

CONTEXT:
{context_text}

QUESTION:
{question}

RESPONSE INSTRUCTION:
- Answer concisely in English (2–6 sentences).
- After your answer, add a line "Sources:" listing the chunks you used, formatted as:
  [book] | title | parwa_id-chunk_id
"""
    else:
        prompt = f"""
Anda adalah asisten yang ahli dalam teks Asta Dasa Parwa (bagian Mahabharata).
Gunakan **hanya** konteks berikut untuk menjawab pertanyaan. Jangan tambahkan fakta di luar konteks.
Jika jawaban tidak ditemukan, jawab:
"Maaf, saya tidak menemukan informasi yang relevan di sumber yang tersedia."

KONTEKS:
{context_text}

PERTANYAAN:
{question}

INSTRUKSI JAWABAN:
- Jawab dalam Bahasa Indonesia (2–6 kalimat) dengan nada informatif.
- Setelah jawaban, sertakan 'Sumber:' dan cantumkan referensi chunk dalam format:
  [book] | judul | parwa_id-chunk_id
"""

    return prompt.strip()


# ============================================================
# MAIN FUNCTION
# ============================================================
def answer_question(question: str, top_k: int = DEFAULT_TOP_K):
    """
    Pipeline utama RAG:
    1. Ambil kandidat FAISS
    2. Rerank pakai cosine
    3. Bangun prompt
    4. Panggil Gemini (2.5 Flash)
    """
    try:
        load_store()
        candidates = retrieve_candidates(question, top_k)
        retrieved = rerank(question, candidates, top_k) if candidates else []

        prompt = build_prompt(question, retrieved)
        response_text = generate_with_gemini(prompt)

        # fallback: jika Gemini gagal memberi text
        if not response_text or response_text.strip() == "":
            response_text = "(Maaf, model tidak memberikan respons yang valid. Silakan coba lagi.)"

        return {"answer": response_text, "retrieved": retrieved}

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return {"answer": "Terjadi kesalahan internal pada sistem RAG.", "retrieved": []}
