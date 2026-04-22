# ingest.py
import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
import faiss
from utils import clean_text, chunk_text

load_dotenv()

# ============ Konfigurasi dasar ============
MYSQL_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@127.0.0.1:3306/astadasaparwa")
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan di .env")

client = genai.Client(api_key=GEMINI_API_KEY)
EMBED_MODEL_NAME = "gemini-embedding-001" 
print(f"🔹 Using NEW GEMINI SDK model: {EMBED_MODEL_NAME}")

# File output
CHUNKS_JSON = os.path.join(DATA_DIR, "parwa_chunks.json")
VECTORS_NPY = os.path.join(DATA_DIR, "parwa_vectors.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH = os.path.join(DATA_DIR, "index_metadata.json")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint_vectors.npy") # 👈 File Save Point

# ============ Fungsi utilitas ============
def fetch_parwa():
    engine = create_engine(MYSQL_URL)
    with engine.connect() as conn:
        q = text("SELECT id, book, sub_parva, section, judul, isi FROM Parwa")
        res = conn.execute(q)
        return [dict(r) for r in res.mappings().all()]

def build_index(vectors, index_path):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, index_path)

def get_gemini_embeddings(texts, batch_size=10):
    all_embeddings = []
    start_idx = 0
    
    # 👈 Cek apakah ada Save Point dari percobaan sebelumnya
    if os.path.exists(CHECKPOINT_FILE):
        all_embeddings = np.load(CHECKPOINT_FILE).tolist()
        start_idx = len(all_embeddings)
        print(f"🔄 CHECKPOINT DITEMUKAN! Melanjutkan progress dari data ke-{start_idx}...")

    for i in range(start_idx, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL_NAME,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                for emb in response.embeddings:
                    all_embeddings.append(emb.values)
                    
                print(f"✅ Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks...")
                
                # 👈 Simpan progress ke Save Point setiap kali sukses 1 batch
                np.save(CHECKPOINT_FILE, np.array(all_embeddings, dtype=np.float32))
                
                time.sleep(3) 
                break 
                
            except Exception as e:
                error_msg = str(e)
                # SEKARANG KITA TAMBAHKAN 503 DAN UNAVAILABLE KE DALAM DAFTAR SABAR
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg or "UNAVAILABLE" in error_msg:
                    print(f"⚠️ Server Google sibuk/kena limit! Istirahat 60 detik... (Percobaan {attempt+1}/{max_retries})")
                    time.sleep(60) 
                else:
                    print(f"❌ Error fatal di batch {i}: {e}")
                    raise e
        else:
            print("\n❌ GAGAL TOTAL: Limit harian API Key ini sepertinya sudah habis.")
            print("Silakan ganti GEMINI_API_KEY di .env dengan akun lain, lalu jalankan ulang script ini.")
            return None # Berhenti dengan aman
            
    return np.array(all_embeddings, dtype=np.float32)

# ============ Pipeline utama ============
def main():
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

    print(f"🧩 Total chunks: {len(all_chunks)} — mengirim ke Gemini API...")
    texts = [c["text_en"] for c in all_chunks]
    
    vectors_np = get_gemini_embeddings(texts)
    
    # Jika gagal karena limit, jangan lanjutkan membuat index (agar file tidak rusak)
    if vectors_np is None:
        return

    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_np = vectors_np / norms

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    np.save(VECTORS_NPY, vectors_np)
    build_index(vectors_np, INDEX_PATH)

    print(f"💾 Saved {len(all_chunks)} chunks to {CHUNKS_JSON}")
    print(f"💾 Saved vectors to {VECTORS_NPY}")
    print(f"💾 Saved FAISS index to {INDEX_PATH}")

    meta = {
        "embed_model": EMBED_MODEL_NAME,
        "num_chunks": len(all_chunks),
        "vector_dimension": vectors_np.shape[1]
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"🧠 Metadata saved to {META_PATH}")

    # Bersihkan file checkpoint jika sudah 100% selesai
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        
    print("✅ Proses indexing menggunakan SDK Baru SELESAI 100%!")

if __name__ == "__main__":
    main()