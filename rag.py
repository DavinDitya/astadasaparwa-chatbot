# rag.py (final, lengkap)
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from gemini_client import generate_with_gemini
from dotenv import load_dotenv

# optional lang detect (pip install langdetect)
try:
    from langdetect import detect
except Exception:
    def detect(_):
        return "id"

load_dotenv()

# Paths (tidak diubah)
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")
META_PATH = os.path.join(DATA_DIR, "index_metadata.json")  # optional metadata

# Embedding model (multilingual recommended)
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# instantiate model lazily (this may download model on first use)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# caches
_index = None
_chunks = None
_vectors = None
_book_names = None
_meta_info = {}

# configuration
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", 4))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.28))
CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", 3500))


def load_store():
    """Load index, metadata, vectors; also log model and chunk info."""
    global _index, _chunks, _vectors, _book_names, _meta_info

    # optional metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            try:
                _meta_info = json.load(f)
                print(f"🧠 Index metadata loaded: model={_meta_info.get('embed_model')} | chunks={_meta_info.get('num_chunks')}")
            except Exception as e:
                print(f"⚠️ Gagal membaca metadata: {e}")

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
        print(f"✅ Loaded metadata. Total chunks: {len(_chunks)}")

    if _vectors is None and os.path.exists(VECTORS_NPY):
        _vectors = np.load(VECTORS_NPY)
        norms = np.linalg.norm(_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _vectors = _vectors / norms
        print("✅ Loaded precomputed vectors.")

    if _book_names is None:
        names = set()
        for c in _chunks:
            b = c.get("meta", {}).get("book")
            if b:
                names.add(b.lower())
        _book_names = sorted(list(names))

    # verify consistency
    model_in_meta = _meta_info.get("embed_model")
    if model_in_meta and model_in_meta != EMBED_MODEL_NAME:
        print(f"⚠️ WARNING: Index dibuat dengan model berbeda ({model_in_meta}) dari yang aktif ({EMBED_MODEL_NAME})")

    return _index, _chunks, _vectors


def detect_book_from_query(q: str):
    """
    Jika query menyebut nama parwa (case-insensitive), kembalikan list nama parwa yang cocok.
    """
    global _book_names
    _, chunks, _ = load_store()
    q_low = q.lower()
    if not _book_names:
        return []
    matches = [name for name in _book_names if name in q_low]
    if not matches:
        return []
    # return canonical names from first occurrences in chunks
    canonical = []
    for m in matches:
        for c in chunks:
            b = c.get("meta", {}).get("book")
            if b and b.lower() == m:
                canonical.append(b)
                break
    return list(dict.fromkeys(canonical))


def retrieve_candidates(query: str, top_k: int = DEFAULT_TOP_K, book_filter: List[str] = None) -> List[Tuple[float, int]]:
    """
    Ambil kandidat dari FAISS atau dari subset book_filter bila disediakan.
    Mengembalikan list of (score, index).
    """
    idx, chunks, vectors = load_store()
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # jika ada filter book, coba gunakan precomputed vectors subset
    if book_filter:
        candidate_indices = [i for i, c in enumerate(chunks) if c.get("meta", {}).get("book") in book_filter]
        if candidate_indices:
            if vectors is not None:
                subvecs = vectors[candidate_indices]
                scores = np.dot(subvecs, q_vec)
                order = np.argsort(-scores)[: top_k * 6]
                return [(float(scores[i]), int(candidate_indices[i])) for i in order]
            else:
                # fallback: scan FAISS results and filter
                D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 10)
                results = []
                for score, i in zip(D[0], I[0]):
                    if i < 0:
                        continue
                    if chunks[i].get("meta", {}).get("book") in book_filter:
                        results.append((float(score), int(i)))
                return results

    # default FAISS search (ambil lebih banyak kandidat untuk rerank)
    D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 6)
    results = [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]
    return results


def rerank(query: str, candidates: List[Tuple[float, int]], top_k: int = DEFAULT_TOP_K) -> List[dict]:
    """
    Re-rank kandidat pakai dot product terhadap precomputed vectors (cosine).
    Kembalikan list {'score':..., 'chunk':...}
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

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    seen = set()
    for score, idx in scored:
        if idx in seen:
            continue
        seen.add(idx)
        if score < SCORE_THRESHOLD:
            continue
        chunk = chunks[idx]
        results.append({"score": float(score), "chunk": chunk})
        if len(results) >= top_k:
            break
    return results


def build_context_text(retrieved: List[dict], char_limit: int = CONTEXT_CHAR_LIMIT) -> str:
    """
    Gabungkan teks retrieved sampai batas karakter; sertakan header singkat tiap chunk.
    """
    parts = []
    total = 0
    for r in retrieved:
        meta = r["chunk"].get("meta", {})
        header = f"[{meta.get('book')}] {meta.get('judul') or ''} (parwa_id={r['chunk'].get('parwa_id')}, chunk={r['chunk'].get('chunk_id')})"
        body = r["chunk"].get("text_en", "")
        snippet = f"{header}\n{body}"
        if total + len(snippet) > char_limit:
            remain = max(0, char_limit - total - 20)
            if remain <= 0:
                break
            snippet = snippet[:remain] + " ... (terpotong)"
            parts.append(snippet)
            total += len(snippet)
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, retrieved: List[dict]) -> str:
    """
    Bangun prompt untuk Gemini; bahasa disesuaikan dengan deteksi bahasa pada query.
    Sertakan instruksi eksplisit: jawab hanya dari konteks, sertakan Sumber.
    """
    lang = "id"
    try:
        lang = detect(question)
    except Exception:
        lang = "id"

    if not retrieved:
        if lang.startswith("en"):
            return f"""You are an expert assistant for the Asta Dasa Parwa (Mahabharata) corpus.
User question:
{question}

I could not find any relevant passages in the provided sources. Please answer briefly:
"I'm sorry, I couldn’t find relevant information in the available sources."
""".strip()
        else:
            return f"""Anda adalah asisten ahli untuk teks Asta Dasa Parwa (Mahabharata).
Pertanyaan pengguna:
{question}

Saya tidak menemukan konteks yang relevan dalam koleksi sumber. Jawab singkat:
"Maaf, saya tidak menemukan informasi yang relevan di sumber yang tersedia."
""".strip()

    context_text = build_context_text(retrieved)
    if lang.startswith("en"):
        prompt = f"""
You are an expert assistant on the Asta Dasa Parwa (Mahabharata) texts.
Use **only** the CONTEXT below to answer the question. Do not invent facts not present in context.
If the answer is not in the context, say: "I'm sorry, I couldn’t find relevant information in the available sources."

CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
- Answer concisely in English (2–6 sentences).
- After the answer, add a line "Sources:" and list used chunks in format:
  [book] | title | parwa_id-chunk_id
"""
    else:
        prompt = f"""
Anda adalah asisten yang ahli dalam teks Asta Dasa Parwa (Mahabharata).
Gunakan **hanya** KONTEKS berikut untuk menjawab pertanyaan. Jangan tambahkan informasi yang tidak ada di konteks.
Jika jawaban tidak ada di konteks, jawab: "Maaf, saya tidak menemukan informasi yang relevan di sumber yang tersedia."

KONTEKS:
{context_text}

PERTANYAAN:
{question}

INSTRUKSI:
- Jawab dalam Bahasa Indonesia ringkas (2–6 kalimat).
- Setelah jawaban, tambahkan baris "Sumber:" lalu cantumkan chunk yang digunakan dalam format:
  [book] | judul | parwa_id-chunk_id
"""
    return prompt.strip()


def answer_question(question: str, top_k: int = DEFAULT_TOP_K):
    """
    Pipeline utama RAG:
     1) load store
     2) detect book filter
     3) retrieve candidates
     4) rerank
     5) build prompt
     6) call Gemini
    """
    try:
        load_store()
        # boost/filter by book mention
        books = detect_book_from_query(question)
        candidates = retrieve_candidates(question, top_k=top_k, book_filter=books or None)
        if not candidates:
            candidates = retrieve_candidates(question, top_k=top_k, book_filter=None)
        retrieved = rerank(question, candidates, top_k=top_k) if candidates else []

        prompt = build_prompt(question, retrieved)
        response_text = generate_with_gemini(prompt)

        if not response_text or isinstance(response_text, (dict, list)) and not str(response_text).strip():
            response_text = "(Model tidak memberikan respons yang valid.)"

        return {"answer": response_text, "retrieved": retrieved}
    except Exception as e:
        print("[RAG ERROR]", e)
        return {"answer": "Terjadi kesalahan internal pada sistem RAG.", "retrieved": []}
