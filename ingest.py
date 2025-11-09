# ingest.py
import os
import json
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import faiss
from utils import clean_text, chunk_text

load_dotenv()

# ============ Konfigurasi dasar ============
MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:@127.0.0.1:3306/astadasaparwa")
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print(f"🔹 Using embedding model: {EMBED_MODEL_NAME}")

# File output
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "index_metadata.json")

# ============ Fungsi utilitas ============
def confirm_overwrite(path):
    if os.path.exists(path):
        print(f"⚠️ File '{path}' sudah ada dan akan ditimpa.")
        return True
    return False

def fetch_parwa():
    engine = create_engine(MYSQL_URL)
    with engine.connect() as conn:
        q = text("SELECT id, book, sub_parva, section, judul, isi FROM parwa")
        res = conn.execute(q)
        return [dict(r) for r in res.mappings().all()]

def build_index(vectors, index_path):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, index_path)

# ============ Pipeline utama ============
def main():
    # Bersihkan file lama jika ada
    for path in [CHUNKS_JSON, VECTORS_NPY, INDEX_PATH, META_PATH]:
        if confirm_overwrite(path):
            os.remove(path)
            print(f"🧹 Menghapus file lama: {path}")

    print("📡 Mengambil data dari database...")
    parwas = fetch_parwa()
    print(f"✅ Fetched {len(parwas)} Parwa rows from DB")

    all_chunks = []
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
            all_chunks.append({
                "parwa_id": pid,
                "chunk_id": i,
                "text_en": ch,
                "meta": meta_base
            })

    print(f"🧩 Total chunks: {len(all_chunks)} — membuat embedding...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text_en"] for c in all_chunks]
    vectors_np = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalisasi vektor
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_np = vectors_np / norms

    # Simpan metadata dan index
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    np.save(VECTORS_NPY, vectors_np)
    build_index(vectors_np, INDEX_PATH)

    print(f"💾 Saved {len(all_chunks)} chunks to {CHUNKS_JSON}")
    print(f"💾 Saved vectors to {VECTORS_NPY}")
    print(f"💾 Saved FAISS index to {INDEX_PATH}")

    # Simpan metadata info
    meta = {
        "embed_model": EMBED_MODEL_NAME,
        "num_chunks": len(all_chunks)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"🧠 Metadata saved to {META_PATH}")

    print("✅ Proses indexing selesai!")

if __name__ == "__main__":
    main()
