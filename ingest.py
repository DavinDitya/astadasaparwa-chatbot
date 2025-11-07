# ingest.py
import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from utils import clean_text, chunk_text

load_dotenv()

# Environment variables (set di .env)
# Example: MYSQL_URL=mysql+pymysql://root:@127.0.0.1:3306/astadasaparwa
MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/astadasaparwa")
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Embedding model (local) — small, fast
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print("Loading embedding model:", EMBED_MODEL_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def fetch_parwa():
    engine = create_engine(MYSQL_URL)
    with engine.connect() as conn:
        q = text("SELECT id, book, sub_parva, section, judul, isi FROM parwa")
        res = conn.execute(q)
        rows = [dict(r) for r in res.mappings().all()]
    return rows

def build_index(chunks_meta, vectors, index_path):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via inner product when vectors normalized
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, index_path)

def main():
    parwas = fetch_parwa()
    print(f"Fetched {len(parwas)} Parwa rows from DB")

    all_chunks = []
    vectors = []

    for p in parwas:
        pid = p["id"]
        meta_base = {
            "id": pid,
            "book": p.get("book"),
            "sub_parva": p.get("sub_parva"),
            "section": p.get("section"),
            "judul": p.get("judul"),
        }
        text_raw = p.get("isi") or ""
        cleaned = clean_text(text_raw)
        chunks = chunk_text(cleaned, max_chars=1500, overlap=200)
        for i, ch in enumerate(chunks, start=1):
            chunk_obj = {
                "parwa_id": pid,
                "chunk_id": i,
                "text_en": ch,
                "meta": meta_base,
            }
            all_chunks.append(chunk_obj)

    print(f"Total chunks: {len(all_chunks)}. Creating embeddings...")

    texts = [c["text_en"] for c in all_chunks]
    vectors_np = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize for inner product similarity
    import numpy as np
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_np = vectors_np / norms

    # Save JSON metadata and texts
    json_path = os.path.join(DATA_DIR, "parwa_chunks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print("Saved chunks metadata to", json_path)

    # Build FAISS index
    index_path = os.path.join(DATA_DIR, "faiss_index.bin")
    d = vectors_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors_np)
    faiss.write_index(index, index_path)
    print("Saved FAISS index to", index_path)

    # Save vectors as numpy for reference
    npy_path = os.path.join(DATA_DIR, "parwa_vectors.npy")
    np.save(npy_path, vectors_np)
    print("Saved vectors to", npy_path)

if __name__ == "__main__":
    main()
