# gemini_client.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

def generate_with_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    """
    Mengirim prompt ke Gemini. Mengembalikan string jawaban, 
    ATAU mengembalikan None jika terkena Safety Filter (agar bisa di-retry).
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Matikan semua sensor
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

        if not response.candidates:
            return None # Gagal total

        candidate = response.candidates[0]

        # Cek Finish Reason
        # 2 = Safety, 3 = Recitation (Hak Cipta/Sitasi)
        if candidate.finish_reason in [2, 3]: 
            print(f"⚠️ Terkena Safety Filter (Reason: {candidate.finish_reason}).")
            # Coba ambil teks parsial jika ada (kadang Gemini ngasih setengah jalan)
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text = "".join([p.text for p in candidate.content.parts if p.text]).strip()
                if text: return text
            
            return None # Kembalikan None agar rag.py melakukan Fallback

        # Ambil teks normal
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            return "".join([part.text for part in candidate.content.parts if part.text]).strip()

        return None

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None