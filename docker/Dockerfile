# CryptoAI Trading System
# Multi-stage build for optimized production image

# Stage 1: Build environment
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml /tmp/
WORKDIR /tmp

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e .


# Stage 2: Runtime environment
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app directory
WORKDIR /app

# Copy application code
COPY cryptoai/ /app/cryptoai/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/checkpoints

# Set permissions
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# Default command
CMD ["python", "-m", "cryptoai.main"]


# Stage 3: Development environment
FROM runtime AS development

# Install development dependencies
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    ipython \
    jupyter

# Mount points for development
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Development command
CMD ["python", "-m", "ipython"]
