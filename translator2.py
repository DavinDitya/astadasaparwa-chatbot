import time
import mysql.connector
from deep_translator import GoogleTranslator

# Koneksi Database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="", 
    database="astadasaparwa" 
)
cursor = db.cursor(dictionary=True)

# Google Translate punya limit karakter per request (sekitar 5000)
# Jadi kita harus pecah kecil-kecil
MAX_CHUNK_SIZE = 4000 

def split_text(text, max_limit=MAX_CHUNK_SIZE):
    """Memecah teks panjang menjadi potongan < 4000 karakter"""
    chunks = []
    while len(text) > max_limit:
        # Cari spasi terdekat sebelum limit agar tidak memotong kata
        split_index = text.rfind(' ', 0, max_limit)
        if split_index == -1:
            split_index = max_limit
        
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks

def start_google_translate():
    translator = GoogleTranslator(source='en', target='id')
    
    print("🚀 Memulai Google Translate (Tanpa API Key)...")

    while True:
        # Ambil 1 data
        cursor.execute("SELECT id, isi FROM Parwa WHERE isi_id IS NULL OR isi_id = '' LIMIT 1")
        row = cursor.fetchone()

        if not row:
            print("🎉 Selesai! Semua data sudah diterjemahkan.")
            break

        text_id = row['id']
        original_text = row['isi']
        print(f"🔄 Memproses ID {text_id} ({len(original_text)} chars)...")

        try:
            # 1. Pecah teks besar jadi kecil-kecil
            chunks = split_text(original_text)
            translated_chunks = []

            for i, chunk in enumerate(chunks):
                # Translate
                res = translator.translate(chunk)
                translated_chunks.append(res)
                
                # Jeda sedikit biar IP tidak di-ban Google
                time.sleep(1) 

            # 2. Gabungkan dan Simpan
            final_text = " ".join(translated_chunks)
            
            cursor.execute("UPDATE Parwa SET isi_id = %s WHERE id = %s", (final_text, text_id))
            db.commit()
            print(f"✅ ID {text_id} Berhasil.")

        except Exception as e:
            print(f"❌ Error ID {text_id}: {e}")
            time.sleep(10) # Kalau error, istirahat agak lama

if __name__ == "__main__":
    start_google_translate()