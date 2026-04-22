import urllib.request
import os

print("🚀 Memulai Bypass Unduhan Database AI via Python...")

# Pastikan folder data ada
os.makedirs("data", exist_ok=True)

# Link raw GitHub untuk file LFS kamu
urls = {
    "data/faiss_index.bin": "https://github.com/DavinDitya/astadasaparwa-chatbot/raw/main/data/faiss_index.bin",
    "data/parwa_vectors.npy": "https://github.com/DavinDitya/astadasaparwa-chatbot/raw/main/data/parwa_vectors.npy"
}

for path, url in urls.items():
    print(f"⬇️ Menyedot {path} dari GitHub...")
    urllib.request.urlretrieve(url, path)
    print(f"✅ Sukses menimpa {path} dengan file asli!")

print("🎉 Semua database berhasil diamankan! Menyalakan Server Utama...")