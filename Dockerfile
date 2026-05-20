FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# MuJoCo headless rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Avoid native thread-related crashes / oversubscription
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

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
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

CMD ["/bin/bash"]