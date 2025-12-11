import os
import json
import time
import mysql.connector
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Gunakan gemini-2.0-flash karena limit free tier lebih besar (1500/day) & stabil
model = genai.GenerativeModel("gemini-2.0-flash") 

# Koneksi Database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="", # Sesuaikan
    database="astadasaparwa" # Sesuaikan
)
cursor = db.cursor(dictionary=True)

SLEEP_TIME = 15  # Istirahat 15 detik agar aman dari limit RPM
BATCH_SIZE = 1 # Tetap 1 agar aman


MAX_CHUNK_SIZE = 8000 # Turunkan ke 8k chars (Turtle Mode) agar token per menit aman

def split_text_smartly(text, max_limit=MAX_CHUNK_SIZE):
    """Memecah teks panjang menjadi chunk berdasarkan paragraf."""
    chunks = []
    current_chunk = ""
    
    # Split berdasarkan paragraf (2 newline)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # Jika satu paragraf saja sudah terlalu besar
        if len(para) > max_limit:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(para) 
            continue
            
        if len(current_chunk) + len(para) + 2 < max_limit:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += para
        else:
            chunks.append(current_chunk)
            current_chunk = para
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def translate_with_retry(prompt, retries=5):
    """Mencoba translate dengan retry logic kalau kena limit (429)."""
    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 8192
                }
            )
            return response.text.strip()
            
        except Exception as e:
            if "429" in str(e):
                wait_time = 60 # Default wait
                print(f"⚠️ Kena Limit (429). Tunggu {wait_time} detik sebelum retry ke-{attempt+1}...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ Error lain: {e}")
                return None
    return None

def translate_text_part(text):
    """Menerjemahkan satu potongan teks dengan Output Text Biasa (Non-JSON)."""
    if not text.strip(): return ""
    
    prompt = f"""
    You are an expert translator for classical literature (Mahabharata).
    
    Task: Translate the following text segment from English to Indonesian.
    Style: Formal, Literary (Sastra), and Epic. Use 'Bahasa Indonesia yang baku'.
    Keep the formatting/newlines approximately the same.
    
    IMPORTANT: Return ONLY the translated text. Do not add any opening words like "Here is the translation" or markdown formatting.
    
    Text to translate:
    {text}
    """
    
    return translate_with_retry(prompt)

def translate_batch(batch_data):
    """
    Versi baru: Memproses input satu per satu dan melakukan chunking jika perlu.
    """
    final_results = []
    
    for item in batch_data:
        text_id = item['id']
        original_text = item['text']
        total_len = len(original_text)
        
        translated_segments = []
        is_failed = False

        # Cek apakah perlu di-chunk
        if total_len > MAX_CHUNK_SIZE:
            print(f"🔹 Mendeteksi teks panjang ({total_len} chars) untuk ID {text_id}. Memecah menjadi beberapa bagian...")
            chunks = split_text_smartly(original_text)
            print(f"   👉 Total {len(chunks)} chunks akan diproses.")
            
            for i, chunk in enumerate(chunks):
                print(f"   ⏳ Memproses chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                t_res = translate_text_part(chunk)
                
                if t_res is None:
                     print(f"❌ Gagal translate chunk {i+1}. Membatalkan simpan untuk ID {text_id} agar bisa diulang nanti.")
                     is_failed = True
                     break
                
                translated_segments.append(t_res)
                print("   🐢 Turtle Mode: Menunggu 30 detik sebelum chunk berikutnya...")
                time.sleep(30) # Jeda PENTING untuk TPM limit
            
            if is_failed:
                continue # Skip ke item berikutnya, jangan simpan yg ini

            # Gabungkan kembali
            full_translation = "\n\n".join(translated_segments)
            
        else:
            # Jika kecil, langsung saja
            full_translation = translate_text_part(original_text)
            if full_translation is None:
                print(f"❌ Gagal translate ID {text_id}. Skip.")
                continue

        final_results.append({"id": text_id, "translation": full_translation})
        
    return final_results

def start_batch_job():
    print("🚀 Memulai Batch Translation...")

    while True:
        # 1. Ambil 10 data yang belum ada bahasa Indonesianya
        cursor.execute(f"SELECT id, isi FROM Parwa WHERE isi_id IS NULL OR isi_id = '' LIMIT {BATCH_SIZE}")
        rows = cursor.fetchall()

        if not rows:
            print("🎉 Semua data sudah diterjemahkan!")
            break

        # 2. Siapkan payload untuk Gemini
        payload = []
        ids_in_batch = []
        for row in rows:
            # Bersihkan sedikit teksnya biar JSON valid
            clean_text = row['isi'].replace('"', "'").strip()
            payload.append({"id": row['id'], "text": clean_text})
            ids_in_batch.append(row['id'])

        print(f"🔄 Memproses Batch ID: {ids_in_batch[0]} s/d {ids_in_batch[-1]} ...")

        # 3. Panggil Gemini
        results = translate_batch(payload)

        # 4. Update Database
        if results:
            updates = 0
            for item in results:
                t_id = item.get("id")
                t_text = item.get("translation")
                
                if t_id and t_text:
                    cursor.execute("UPDATE Parwa SET isi_id = %s WHERE id = %s", (t_text, t_id))
                    updates += 1
            
            db.commit()
            print(f"✅ Berhasil update {updates} data.")
        else:
            print("❌ Gagal translate batch ini. Melompati sementara...")
            # Opsional: Tandai error biar gak looping terus, tapi untuk sekarang biarkan
        
        # 5. Jeda Waktu (PENTING untuk Free Tier)
        print(f"⏳ Istirahat {SLEEP_TIME} detik...")
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    try:
        start_batch_job()
    except KeyboardInterrupt:
        print("\n🛑 Proses dihentikan manual.")
    finally:
        db.close()