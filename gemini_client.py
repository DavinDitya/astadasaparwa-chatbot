# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Muat variabel dari file .env
load_dotenv()

# Ambil API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

# Konfigurasi Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pilih model Gemini (default: Gemini 2.5 Flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

def generate_with_gemini(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """
    Mengirim prompt ke Gemini API dengan fallback aman untuk berbagai struktur respons.
    Cocok untuk RAG pipeline (panjang konteks + jawaban ringkas & relevan).
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            [prompt],
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40,
            },
            safety_settings=None,  # hindari false-positive block
        )

        # 1️⃣ Ambil hasil utama
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # 2️⃣ Fallback: ambil dari candidates/parts
        if hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                if cand.content and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if part.text and part.text.strip():
                            return part.text.strip()

        # 3️⃣ Fallback terakhir jika semua kosong
        return "(Tidak ada respons teks yang valid dari Gemini)"

    except Exception as e:
        # Tangani error generik dan log
        print(f"[Gemini Error] {e}")
        if "429" in str(e):
            return "(Server Gemini membatasi permintaan sementara. Coba lagi nanti.)"
        elif "safety" in str(e).lower():
            return "(Jawaban diblokir oleh sistem keamanan model.)"
        else:
            return f"(Terjadi kesalahan: {e})"
