FROM python:3.12-slim

WORKDIR /app

# System deps for implicit (BLAS/LAPACK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (no torch for smaller image)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend
COPY config/ config/
COPY src/ src/
COPY data/processed/ data/processed/
COPY data/embeddings/ data/embeddings/
COPY models/collaborative/als_user_factors.npy models/collaborative/
COPY models/collaborative/als_item_factors.npy models/collaborative/
COPY models/collaborative/als_matrix.npz models/collaborative/
COPY models/collaborative/als_meta.npz models/collaborative/
COPY models/graph/ models/graph/
COPY run.py .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
