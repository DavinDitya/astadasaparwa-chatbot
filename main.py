# main.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from rag import answer_question, load_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asta-dasa-chatbot")

app = FastAPI(title="Asta Dasa Chatbot (RAG with Gemini)")

class AskRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    top_k: int = 10
    # [BARU] Tambahkan mode panjang/singkat
    mode: str = "detail" # Defaultnya "detail", bisa diisi "singkat" dari Android

@app.on_event("startup")
async def startup_event():
    try:
        load_store()
        logger.info("✅ FAISS index dan metadata berhasil dimuat di startup.")
    except Exception as e:
        logger.error(f"Gagal memuat index: {e}")

@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question or req.question.strip() == "":
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
    try:
        # [UPDATE] Kirim mode ke fungsi rag
        result = answer_question(req.question, history=req.history, top_k=req.top_k, mode=req.mode)
        return result
    except Exception as e:
        logger.error(f"Error dalam pemrosesan pertanyaan: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan di server chatbot.")