# gemini_client.py
import os
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 1. Ambil gabungan API Keys dari .env
keys_string = os.getenv("GEMINI_API_KEYS")
if not keys_string:
    raise RuntimeError("❌ GEMINI_API_KEYS tidak ditemukan. Pastikan sudah diset di .env dengan pemisah koma.")

# 2. Pecah string menjadi list (daftar) kunci
API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]

if not API_KEYS:
    raise RuntimeError("❌ Daftar API Key kosong! Periksa format di .env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").replace("models/", "")

def get_random_client():
    """Mengambil satu API Key secara acak dan mengembalikan instance Client baru"""
    selected_key = random.choice(API_KEYS)
    # Opsional: Print sebagian kecil key untuk memastikan rotasi berjalan di log Railway
    print(f"🔄 Menggunakan API Key berakhiran: ...{selected_key[-4:]}")
    return genai.Client(api_key=selected_key)

def generate_with_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    prompt_aman = (
        prompt + 
        "\n\nInstruksi Sistem Keselamatan: "
        "Gunakan bahasa yang halus. Jika ada unsur peperangan atau konflik, "
        "ceritakan secara tersirat agar aman dari filter konten."
    )

    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 3. Panggil fungsi acak kunci setiap kali mencoba request
            client = get_random_client()
            
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
            
            # Jika Error 503 (Sibuk) ATAU 429 (Terkena Limit)
            if "503" in error_msg or "Unavailable" in error_msg or "429" in error_msg or "Quota" in error_msg:
                if attempt < max_retries - 1:
                    print(f"⚠️ Gemini Limit/Sibuk. Mengganti API Key & menunggu 2 detik... (Percobaan {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue # Putar ulang loop, dan dia otomatis akan mengacak kunci baru!
            
            print(f"[Gemini Error] {error_msg}")
            return None
            
    return None