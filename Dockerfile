# Use the official PyTorch image with CUDA 11.8 and cuDNN 9
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Install Python 3.11 (already included in the PyTorch image, but ensuring latest version)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    libgl1-mesa-glx \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    libclang-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

# Install other Python requirements
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "src/agent/train_single.py"]
