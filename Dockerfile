# ── Stage 1: builder ──────────────────────────────────────────────────────────
# This stage installs all build tools and Python dependencies.
# It produces a /opt/venv with everything installed.
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed to compile some Python packages (e.g. tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch first so pip doesn't pull the huge CUDA build.
# This keeps the final image around 1.2 GB instead of ~3 GB.
RUN pip install --no-cache-dir \
    "torch>=2.0.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Copy dependency list and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and install the mtnsails CLI entry point
COPY . .
RUN pip install --no-cache-dir -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Smaller final image — only the runtime essentials.
# Build tools from Stage 1 are NOT included here.
FROM python:3.11-slim AS runtime

# libgomp1 is required by PyTorch/ONNX Runtime for multi-threaded CPU ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment and source from the builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /app

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# HuggingFace will store downloaded models here.
# This path is mapped to a persistent named volume (see docker-compose.yml)
# so the ~500 MB Qwen model is only downloaded once.
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Default runtime settings — override with -e or docker compose environment:
#   MTNSAILS_EPOCHS=10 podman compose run train
ENV MTNSAILS_MODEL_NAME=Qwen/Qwen2.5-0.5B
ENV MTNSAILS_EPOCHS=3
ENV MTNSAILS_BATCH_SIZE=4

# The container behaves like the mtnsails CLI command.
# Running `podman compose run train` passes "train ..." as CMD arguments.
ENTRYPOINT ["mtnsails"]
CMD ["--help"]
