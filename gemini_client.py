# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# === Muat variabel dari .env ===
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash") # Gunakan model terbaru jika bisa, atau 1.5-flash

def generate_with_gemini(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """
    Mengirim prompt ke Gemini API dengan pengaturan keamanan MINIMAL
    agar konten sastra (perang/konflik) tidak terblokir.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # ⚠️ SETTING KUNCI: MATIKAN SEMUA BLOCKING
        # Ini memberitahu Gemini: "Jangan blokir apapun, tampilkan saja walau ada kata kekerasan"
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
            safety_settings=safety_settings,
        )

        # === PARSING HASIL ===
        if not response.candidates:
            return "Maaf, sistem sedang sibuk. Coba lagi nanti."

        candidate = response.candidates[0]

        # Cek jika masih kena filter (jarang terjadi jika BLOCK_NONE)
        if candidate.finish_reason == 2: # Safety
            print("⚠️ Terkena Safety Filter padahal sudah BLOCK_NONE.")
            # Tetap coba ambil teksnya jika ada (kadangkala ada partial text)
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                 return candidate.content.parts[0].text
            return "Maaf, saya tidak bisa menjawab pertanyaan ini karena kebijakan konten Google."

        # Ambil teks normal
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            return "".join([part.text for part in candidate.content.parts if part.text]).strip()

        return "Maaf, terjadi kesalahan saat memproses jawaban."

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Maaf, terjadi gangguan koneksi ke otak AI."