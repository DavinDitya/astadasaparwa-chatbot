# gemini_client.py
import os
import time  # <--- [PERBAIKAN] Tambah import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY tidak ditemukan. Pastikan sudah diset di .env")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").replace("models/", "")

def generate_with_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    # --- [PERBAIKAN] Hapus paksaan "Ceritakan secara lengkap, detail" ---
    prompt_aman = (
        prompt + 
        "\n\nInstruksi Sistem Keselamatan: "
        "Gunakan bahasa yang halus. Jika ada unsur peperangan atau konflik, "
        "ceritakan secara tersirat agar aman dari filter konten."
    )

    max_retries = 3 # --- [PERBAIKAN] Kita coba 3 kali jika server sibuk

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_aman,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    safety_settings=[
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    ]
                )
            )

            if not response.candidates:
                return None

            candidate = response.candidates[0]

            if candidate.finish_reason and candidate.finish_reason.name == "SAFETY":
                print(f"⚠️ Terkena Safety Filter (Reason: {candidate.finish_reason.name}).")
                return "Maaf, bagian cerita ini memiliki unsur konflik/peperangan yang diblokir oleh sistem keamanan AI Google. Cobalah bertanya dari sudut pandang nilai moralnya."

            if response.text:
                return response.text.strip()

            return None

        except Exception as e:
            error_msg = str(e)
            # --- [PERBAIKAN] Logika Tunda 2 Detik Jika 503 ---
            if "503" in error_msg or "Unavailable" in error_msg:
                if attempt < max_retries - 1:
                    print(f"⚠️ Gemini 503 (Sibuk). Menunggu 2 detik untuk coba lagi... (Percobaan {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue # Putar ulang loop
            
            print(f"[Gemini Error] {error_msg}")
            return None
            
    return None