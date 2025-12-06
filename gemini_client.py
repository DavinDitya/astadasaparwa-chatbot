# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# === Muat variabel dari .env ===
load_dotenv()

# Ambil API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

# Konfigurasi Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pilih model (default: Gemini 2.5 Flash)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# === Fungsi utama ===
def generate_with_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Mengirim prompt ke Gemini API dan mengembalikan teks hasilnya.
    Didesain untuk menangani safety blocks (finish_reason: 2) 
    tanpa crash, dengan tidak pernah mengakses response.text.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ]

        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
            safety_settings=safety_settings,
        )

        # === PARSING SUPER AMAN ===
        # JANGAN PERNAH gunakan response.text, itu adalah "perangkap".
        # Langsung periksa candidates.

        if not response.candidates:
            # Jika tidak ada kandidat sama sekali
            return "(Model tidak memberikan respons kandidat.)"

        # Ambil kandidat pertama (biasanya hanya ada satu)
        candidate = response.candidates[0]

        # Periksa finish_reason SEBELUM mencoba mengambil teks
        if candidate.finish_reason == 2: # 2 = SAFETY
            print("⚠️ Respons diblokir oleh sistem keamanan Gemini (finish_reason=2).")
            return "(Tidak ada teks yang dihasilkan — diblokir oleh sistem keamanan Gemini.)"
        
        if candidate.finish_reason == 3: # 3 = RECITATION
            print("⚠️ Respons diblokir karena sitasi (finish_reason=3).")
            return "(Tidak ada teks yang dihasilkan — diblokir karena sitasi.)"

        if candidate.finish_reason != 1: # 1 = STOP
            print(f"⚠️ Respons berhenti dengan alasan tidak terduga: {candidate.finish_reason}")
            return f"(Respons berhenti dengan alasan: {candidate.finish_reason})"

        # --- Jika kita sampai di sini, berarti finish_reason == 1 (STOP) dan aman ---
        
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            texts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    texts.append(part.text)
            
            if texts:
                return "\n".join(texts).strip()
            else:
                # Seharusnya tidak terjadi jika finish_reason=1, tapi untuk jaga-jaga
                return "(Respons berhasil namun tidak mengandung teks.)"
        
        # Fallback jika struktur tidak dikenali
        return "(Struktur respons valid tetapi tidak dapat diekstrak teksnya.)"

    except Exception as e:
        # Ini akan menangkap error lain, tapi SEHARUSNYA tidak lagi menangkap
        # error 'Invalid operation' karena kita tidak lagi memakai response.text
        print(f"[Gemini Error] {e}")
        
        # Kita cek secara spesifik apakah ini error yg kita hindari
        if "Invalid operation" in str(e):
             return "(Terjadi kesalahan kritis: 'Invalid operation' masih terpicu.)"
             
        return f"(Terjadi kesalahan pada Gemini Client: {e})"
