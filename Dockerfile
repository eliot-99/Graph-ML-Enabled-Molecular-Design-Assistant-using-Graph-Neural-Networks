# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.10-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies for RDKit (as root, then switch back)
USER root
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements first for caching
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Force unbuffered Python output for logging
ENV PYTHONUNBUFFERED=1

# Run the Flask app
CMD ["python", "app.py"]
