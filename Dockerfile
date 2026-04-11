FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TDATR_CPU_MODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Pin versions that the code expects
RUN pip install --no-cache-dir 'omegaconf==2.0.6' 'hydra-core==1.0.7'

# Install remaining deps (skip problematic/pinned packages)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    $(grep -v '^torch==' requirements.txt \
    | grep -v '^torchvision' \
    | grep -v 'pycosat' \
    | grep -v 'omegaconf' \
    | grep -v 'hydra-core' \
    | grep -v '^[[:space:]]*$' \
    | sed 's/==.*//' | tr '\n' ' ')

# Add app code
COPY . .

# Download model weights at build time
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('CCWM/TDATR', 'model.pt', local_dir='/app')"

# Test image list (users mount their own)
RUN echo '[]' > /app/test_images.json

ENTRYPOINT ["python", "TDATR/eval/infer.py"]
