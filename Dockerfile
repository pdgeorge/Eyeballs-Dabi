FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip git curl ca-certificates && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
        "transformers>=4.41.1" "tokenizers>=0.19.0" \
        fastapi uvicorn pillow python-multipart

ENV MODEL_NAME=Salesforce/blip-image-captioning-base \
    API_PORT=1000 \
    HF_HOME=/models
WORKDIR /workspace
COPY app.py .

HEALTHCHECK CMD curl -fs http://localhost:${API_PORT}/ping || exit 1
EXPOSE 1000
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${API_PORT}"]
