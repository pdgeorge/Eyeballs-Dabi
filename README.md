# Eyeballs‑Dabi 👀 🗨️  
GPU‑accelerated **image‑to‑text captioning** API powered by Hugging Face Transformers, served through FastAPI & Uvicorn and packaged in a single Docker container.

|                    |                           |
|--------------------|---------------------------|
| **Default model**  | `Salesforce/blip-image-captioning-base` |
| **GPU**            | Designed for any NVIDIA card with ≥ 8 GB (tested on RTX 4070 SUPER) |
| **Port**           | `1000` (configurable) |
| **API style**      | REST (JSON) |
| **License**        | Apache‑2.0 (code) · Model licenses per Hugging Face cards |

---

## ️✨ Features
* **One‑command deploy** – `docker run … eyeballs-dabi` pulls everything, attaches the GPU and starts listening.
* **Model hot‑swap** – change `MODEL_NAME` at container launch without rebuilding.
* **Health check** – `/ping` endpoint returns `{"ok": true}` for easy orchestration probes.
* **VRAM‑friendly** – defaults to BLIP‑base (~1 GB fp16); larger models fit in < 7 GB.
* **Persistent cache** – weights stored in a named Docker volume (`blip-cache`) so restarts are instant.

---

## 🖥️ Prerequisites
1. **WSL 2 or native Linux** with NVIDIA driver ≥ R570 (CUDA 12 runtime).
2. **Docker Engine** with the **NVIDIA Container Toolkit** configured:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.3.0-runtime-ubuntu22.04 nvidia-smi
   ```

# Building and Running

## 2. Building (Clean build)
docker build --no-cache -t eyeballs-dabi .

## 3. Run (eyeballs-dabi on port 1000)
Default, runs BLIP-base

To run:
```
docker run -d \
  --name eyeballs-dabi \
  --gpus "device=0" \
  -e MODEL_NAME=Salesforce/blip-image-captioning-large \
  -e API_PORT=1000 \
  -p 1000:1000 \
  -v blip-cache:/models \
  --shm-size=1g \
  eyeballs-dabi
```

Optional flag:

To change models:

`-e MODEL_NAME= (model name)`

## 4. Stopping and destroying
docker stop eyeballs-dabi

docker rm eyeballs-dabi

# Commands

## Pinging:

`curl -X GET http://localhost:1000/ping`

Expected response: `{"ok":true}`

## Upload a file for captioning

`curl -X POST http://localhost:1000/caption -F "file=@1.png"`

Expected response: `{"caption":"<response of image>"}`