FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MUJOCO_GL=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    xvfb \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libosmesa6 \
    libegl1 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

CMD ["/bin/bash"]