#!/bin/bash
echo "🚀 Mencegat Railway! Mengunduh Database LFS secara paksa..."

# Mengunduh file 142MB langsung menimpa file resi yang rusak
curl -L -o data/faiss_index.bin https://github.com/DavinDitya/astadasaparwa-chatbot/raw/main/data/faiss_index.bin
curl -L -o data/parwa_vectors.npy https://github.com/DavinDitya/astadasaparwa-chatbot/raw/main/data/parwa_vectors.npy

echo "✅ Database asli berhasil diamankan! Menyalakan Server Utama..."

# Menyalakan aplikasi FastAPI
uvicorn main:app --host 0.0.0.0 --port $PORT