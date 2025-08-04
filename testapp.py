#!/usr/bin/env python
"""
Minimal FastAPI image‑caption test.
Start:   python testapp.py --port 1000
Test:    curl -F "file=@your.png" http://localhost:1000/caption
"""

import argparse, io, os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import pipeline
import torch, uvicorn

def build_pipeline(model_name: str) -> "transformers.Pipeline":
    # Load everything on GPU 0; fits easily in fp16 on 4070 SUPER
    return pipeline(
        "image-to-text",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_length=64,
    )

def create_app(model_name: str) -> FastAPI:
    captioner = build_pipeline(model_name)
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.post("/caption")
    async def caption(file: UploadFile = File(...)):
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        text = captioner(image)[0]["generated_text"]
        torch.cuda.empty_cache()
        return {"caption": text}

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base",
                        help="Hugging Face model ID")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port FastAPI should bind to")
    args = parser.parse_args()

    app = create_app(args.model)
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
