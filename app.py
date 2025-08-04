#!/usr/bin/env python3
"""
Eyeballs‑Dabi – Image‑to‑Text Captioning API
-------------------------------------------
• GET  /ping       → {"ok": true}
• POST /caption    (multipart form‑data, field “file”) → {"caption": "…"}

Run directly for quick tests:

    python3 app.py                       # default model, port 1000
    python3 app.py --model Salesforce/blip-image-captioning-large --port 1234
"""

import argparse, io, os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import pipeline, AutoTokenizer
import torch, uvicorn

# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #
def get_config() -> tuple[str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Salesforce/blip-image-captioning-base"),
                        help="Hugging Face model ID")
    parser.add_argument("--port",  type=int, default=int(os.getenv("API_PORT", 1000)),
                        help="Port to bind the HTTP server")
    args, _ = parser.parse_known_args()
    return args.model, args.port

def build_pipeline(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    return pipeline(
        "image-to-text",
        model=model_id,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_length=64,
    )

# --------------------------------------------------------------------------- #
# FastAPI application                                                          #
# --------------------------------------------------------------------------- #
MODEL_NAME, API_PORT = get_config()
captioner = build_pipeline(MODEL_NAME)

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"ok": True}

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    text  = captioner(image)[0]["generated_text"]
    torch.cuda.empty_cache()
    return {"caption": text}

# --------------------------------------------------------------------------- #
# Stand‑alone entry point                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
