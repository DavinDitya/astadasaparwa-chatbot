# gemini_client.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

# Inisialisasi client SDK Baru
client = genai.Client(api_key=GEMINI_API_KEY)

# SDK baru lebih suka nama model tanpa awalan "models/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").replace("models/", "")

def generate_with_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    """
    Mengirim prompt ke Gemini menggunakan SDK google-genai terbaru.
    Mengembalikan string jawaban utuh.
    """
    try:
        # Instruksi rahasia: Cerita panjang, tapi hindari deskripsi kekerasan eksplisit
        prompt_aman = (
            prompt + 
            "\n\nInstruksi Sistem: Ceritakan secara lengkap, detail, dan jangan dipotong. "
            "Namun, gunakan bahasa yang halus. Jika ada unsur peperangan atau konflik, "
            "ceritakan secara tersirat agar aman dari filter konten."
        )

        # Pemanggilan API menggunakan struktur Config SDK Baru
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt_aman,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ]
            )
        )

        if not response.candidates:
            return None

        candidate = response.candidates[0]

        # JIKA TERKENA FILTER: Cek menggunakan sistem Enum yang baru
        if candidate.finish_reason and candidate.finish_reason.name == "SAFETY":
            print(f"⚠️ Terkena Safety Filter (Reason: {candidate.finish_reason.name}).")
            return "Maaf, bagian cerita ini memiliki unsur konflik/peperangan yang diblokir oleh sistem keamanan AI Google. Cobalah bertanya dari sudut pandang nilai moralnya."

        # JIKA AMAN: Ambil teks normal dan utuh
        if response.text:
            return response.text.strip()

        return None

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None