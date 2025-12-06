# rag.py (revisi: normalisasi book, metadata-boost, cleanup)
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

# === Paths (tidak diubah) ===
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")
META_PATH = os.path.join(DATA_DIR, "index_metadata.json")  # optional metadata

# === Embedding model (multilingual recommended) ===
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# instantiate (lazy download on first run if needed)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# === Caches / globals ===
_index = None
_chunks = None
_vectors = None
_book_names = None  # list of lowercased book names
_meta_info = {}

# === Configs ===
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", 4))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.28))
CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", 3500))
METADATA_BOOST_FACTOR = float(os.getenv("RAG_METADATA_BOOST", 1.15))  # 15% boost by default


# -------------------------
# Utilities
# -------------------------
def normalize_book_name(s: str) -> str:
    """
    Normalisasi nama book untuk toleransi 'w' <-> 'v' dan whitespace/case.
    Contoh: "Adi Parwa" -> "adi parva"
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace("w", "v")  # make parwa/parva consistent
    s = " ".join(s.split())
    return s


# optional expansion helper (already in your code; kept)
def expand_question(question: str) -> str:
    q = question.lower().strip()
    expansions = []

    if "isi utama" in q:
        expansions.append("ceritakan ringkasan cerita, tema, dan peristiwa penting dari kitab tersebut")
    if "cerita" in q or "kisah" in q:
        expansions.append("jelaskan alur dan tokoh-tokoh utama yang terlibat")
    if "makna" in q or "arti" in q:
        expansions.append("jelaskan penjelasan filosofis dan nilai moralnya")
    if "siapa" in q:
        expansions.append("sebutkan tokoh atau nama karakter yang relevan dalam cerita")

    if expansions:
        question = question + ". " + ". ".join(expansions)
    return question


# -------------------------
# Load store
# -------------------------
def load_store():
    """
    Load FAISS index, chunks metadata, and precomputed vectors (npy) if ada.
    Also create canonical lowercased book name list for detection.
    """
    global _index, _chunks, _vectors, _book_names, _meta_info

    # optional metadata file (created at ingest time if wanted)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
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
                names.add(normalize_book_name(b))
        _book_names = sorted(list(names))

    # warn if mismatch model
    model_in_meta = _meta_info.get("embed_model")
    if model_in_meta and model_in_meta != EMBED_MODEL_NAME:
        print(f"⚠️ WARNING: Index dibuat dengan model berbeda ({model_in_meta}) dari yang aktif ({EMBED_MODEL_NAME})")

    return _index, _chunks, _vectors


# -------------------------
# Book detection (tolerant)
# -------------------------
def detect_book_from_query(q: str) -> List[str]:
    """
    Detect book names mentioned in query. Return canonical names (as appear in metadata)
    but tolerant to parwa/parva spelling; returns list (possibly empty).
    """
    _, chunks, _ = load_store()
    q_norm = normalize_book_name(q)
    if not _book_names:
        return []

    matches = []
    for bn in _book_names:
        if bn in q_norm:
            # find first canonical occurrence (original case) from chunks
            for c in chunks:
                b = c.get("meta", {}).get("book")
                if b and normalize_book_name(b) == bn:
                    matches.append(b)
                    break
    # deduplicate preserve order
    return list(dict.fromkeys(matches))


# -------------------------
# Retrieval
# -------------------------
def retrieve_candidates(query: str, top_k: int = DEFAULT_TOP_K, book_filter: List[str] = None):
    """
    Retrieve candidate (score, idx) pairs. If book_filter provided, prefer chunks from those books.
    """
    idx, chunks, vectors = load_store()
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # if book_filter requested: try using precomputed vectors subset (fast + accurate)
    if book_filter:
        candidate_indices = [i for i, c in enumerate(chunks) if c.get("meta", {}).get("book") in book_filter]
        if candidate_indices:
            if vectors is not None:
                subvecs = vectors[candidate_indices]
                scores = np.dot(subvecs, q_vec)
                order = np.argsort(-scores)[: top_k * 6]
                return [(float(scores[i]), int(candidate_indices[i])) for i in order]
            else:
                # fallback: search broader FAISS then filter by book
                D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 12)
                results = []
                for score, i in zip(D[0], I[0]):
                    if i < 0:
                        continue
                    if chunks[i].get("meta", {}).get("book") in book_filter:
                        results.append((float(score), int(i)))
                return results

    # default FAISS (get more candidates for rerank)
    D, I = idx.search(np.expand_dims(q_vec, axis=0), top_k * 6)
    return [(float(score), int(i)) for score, i in zip(D[0], I[0]) if i >= 0]


# -------------------------
# Rerank + metadata boost
# -------------------------
def rerank(query: str, candidates: List[Tuple[float, int]], top_k: int = DEFAULT_TOP_K, book_mentions: List[str] = None) -> List[dict]:
    """
    Re-rank candidates using precomputed vectors (cosine). Apply metadata boost when
    a chunk's book matches book_mentions (book detected from query).
    """
    _, chunks, precomputed = load_store()
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # normalize book mentions for comparison
    book_norms = [normalize_book_name(b) for b in (book_mentions or [])]

    scored = []
    for faiss_score, idx in candidates:
        if precomputed is not None and 0 <= idx < precomputed.shape[0]:
            score = float(np.dot(q_vec, precomputed[idx]))
        else:
            score = float(faiss_score)

        # metadata boost: if this chunk's book is mentioned in query, increase score
        book = chunks[idx].get("meta", {}).get("book", "")
        if book and any(normalize_book_name(book) == bn for bn in book_norms):
            score = score * METADATA_BOOST_FACTOR

        scored.append((score, idx))

    # sort & filter by threshold
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


# -------------------------
# Build prompt & context
# -------------------------
def build_context_text(retrieved: List[dict], char_limit: int = CONTEXT_CHAR_LIMIT) -> str:
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


def build_prompt(question: str, retrieved_chunks: list) -> str:
    """
    Membangun prompt yang aman secara moderasi untuk Gemini.
    Fokus pada analisis literatur, bukan narasi eksplisit.
    """

    # PANGGIL FUNGSI YANG SUDAH ANDA BUAT UNTUK MEMBATASI KONTEKS
    context_str = build_context_text(retrieved_chunks)

    # ======== 🔒 PROMPT BARU YANG LEBIH AMAN ========
    prompt = f"""
You are an expert literary researcher specializing in classical Hindu epics.
Your task is to answer the user's question based *only* on the provided context.

Question:
{question}

---
[START CONTEXT]
The following are reference excerpts from an epic text. These texts describe mythological events, 
including conflicts, battles, and death, which must be analyzed from a purely literary and 
academic perspective.

{context_str}

[END CONTEXT]
---

Guidelines:
- Analyze the text academically. 
- Do not reproduce or narrate violent events. Summarize them abstractly (e.g., "it describes a conflict between X and Y" or "it lists the key events of the parva").
- If the question asks for a summary, provide the main ideas, not a story.
- Respond in a calm, academic tone.

Now, please provide your answer based *only* on the context provided above:
"""
    # ====================================================

    return prompt.strip()



# -------------------------
# Main pipeline
# -------------------------
def answer_question(question: str, top_k: int = DEFAULT_TOP_K):
    """
    RAG pipeline:
      1) load_store
      2) expand_question (query expansion)
      3) detect book mentions
      4) retrieve (optionally filter/boost by book)
      5) rerank (with metadata boost)
      6) build prompt + call Gemini
    """
    try:
        load_store()

        # 1. expand query to improve retrieval recall
        expanded_q = expand_question(question)

        # 2. detect book mentions (tolerant)
        books = detect_book_from_query(expanded_q)  # returns list e.g. ['Adi Parva']

        # 3. retrieve candidates (try with book filter first if any)
        candidates = retrieve_candidates(expanded_q, top_k=top_k, book_filter=books or None)
        if not candidates:
            candidates = retrieve_candidates(expanded_q, top_k=top_k)

        # 4. rerank with metadata boost using books detected
        retrieved = rerank(expanded_q, candidates, top_k=top_k, book_mentions=books) if candidates else []
        
        limited_retrieved = retrieved[:1]

        # 5. build prompt & call Gemini
        prompt = build_prompt(question, limited_retrieved)
        response_text = generate_with_gemini(prompt)

        if not response_text or (isinstance(response_text, (dict, list)) and not str(response_text).strip()):
            response_text = "(Model tidak memberikan respons yang valid.)"

        return {"answer": response_text, "retrieved": retrieved}
    except Exception as e:
        print("[RAG ERROR]", e)
        return {"answer": "Terjadi kesalahan internal pada sistem RAG.", "retrieved": []}
