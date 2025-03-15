FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install necessary packages
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0

COPY . /lang-segment-anything

# Install dependencies
WORKDIR /lang-segment-anything
ADD https://rye.astral.sh/get /tmp/rye-install.sh
RUN RYE_INSTALL_OPTION="--yes" bash /tmp/rye-install.sh
ENV PATH="$PATH:/root/.rye/shims"
RUN --mount=target=/root/.cache/uv,type=cache,sharing=locked \
    rye sync --no-lock --no-dev

EXPOSE 8000
VOLUME [ "/root/.cache/torch/hub/checkpoints" ]
VOLUME [ "/root/.cache/huggingface/hub" ]

# Entry point
CMD ["rye", "run", "app"]
