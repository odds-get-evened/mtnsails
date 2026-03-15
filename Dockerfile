FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some Python packages (e.g. tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and install the package
COPY . .
RUN pip install --no-cache-dir -e .

# Redirect HuggingFace cache to a directory we can mount as a volume
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# Create runtime directories
RUN mkdir -p /app/cache /app/data /app/models /app/onnx

# Default: show help; override CMD to run any subcommand
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
