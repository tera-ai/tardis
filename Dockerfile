# NOTE: nvcc --version fails if using the `runtime` base image, instead of the `devel` one, since CUDA toolkit is only included on `devel`
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    git \
    gnupg \
    gpg-agent \
    python3-pip \
    python3.10 \
    python3.10-dev \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install the Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    absl-py \
    chex==0.1.82 \
    datasets \
    dill \
    distrax \
    einops \
    fastapi \
    gcsfs \
    gradio \
    huggingface_hub \
    jupyter_http_over_ws \
    lm-eval \
    matplotlib \
    ml_collections \
    mlxu \
    "numpy<2" \
    optax==0.2.1 \
    pydantic \
    requests \
    scipy==1.12.0 \
    seaborn \
    sentencepiece \
    tensorflow \
    torch \
    tqdm \
    transformers \
    uvicorn \
    wandb

# Install JAX separately
RUN pip install --no-cache-dir \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    flax==0.7.0 \
    jax[cuda11_pip]==0.4.14

# Clone your repository
WORKDIR /app
COPY . /app

# Set up GCS credentials
# TODO: remove this after we find a better way to pass in credentials
COPY application_default_credentials.json ~/.config/gcloud/application_default_credentials.json

# Verify
RUN pip freeze | grep jax
RUN pip freeze | grep flax
RUN nvcc --version

# Log of changes
# 1. Cuda 12.6 (and a thousand other tries)
# 2. Cuda 11.8 -> "AttributeError: module 'jax.config' has no attribute 'define_bool_state'"
#   - https://github.com/google/flax/issues/3180
# 3. Fix JAX and FLAX versions (jax[cuda11_pip]==0.4.14, flax==0.7.0, chex==0.1.82, optax==0.2.1) -> "AttributeError: np.issctype was removed in the NumPy 2.0 release. Use issubclass(rep, np.generic) instead.. Did you mean: 'isdtype'?"
#   - https://github.com/ggerganov/whisper.cpp/issues/2257
# 4. Fix Numpy version ("numpy<2" ~= 1.26.4) -> ?
#   - https://github.com/octo-models/octo/issues/71
# 5. Fix Scipy version (1.13.0 ~= 1.12.0) -> OOM GPU Error
# 6. Use 40GB A100 GPUs for the code with toy dataset -> RuntimeError

# TODO: optax in the repo is still lower (0.1.7) than suggested version from installing jax (0.2.1)
# TODO: other packages whose versions remain to be fixed
# 1. distrax==0.1.4
# 2. transformers==4.31.0
# 3. torch==2.0.1
# 4. huggingface_hub==0.16.4
# 5. datasets==2.14.2
# 6. tensorflow==2.17
# 7. mlxu==0.1.11
