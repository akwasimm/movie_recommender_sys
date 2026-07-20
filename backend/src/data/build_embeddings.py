"""
CineScope BERT Embedding Builder
Encodes movie text features into 384-dim embeddings using all-MiniLM-L6-v2
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bert_builder")

ROOT = Path(__file__).resolve().parent.parent.parent
BATCH_SIZE = 512


def run():
    log.info("=" * 60)
    log.info("CineScope BERT Embedding Builder")
    log.info("=" * 60)

    parquet_path = ROOT / "data" / "processed" / "movies_clean.parquet"
    if not parquet_path.exists():
        log.error(f"Run preprocessor first: {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df)} movies")

    mask = df["text_features"].str.strip().str.len() > 5
    df_valid = df[mask].copy()
    log.info(f"Movies with valid text: {len(df_valid)}")

    texts = df_valid["text_features"].tolist()
    ids = df_valid["tmdb_id"].tolist()

    log.info("Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info(f"Encoding {len(texts)} texts (batch={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    out_dir = ROOT / "data" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "bert_embeddings.npy"), embeddings)
    np.save(str(out_dir / "bert_movie_ids.npy"), np.array(ids))

    log.info(f"Saved embeddings: {embeddings.shape}")
    log.info(f"Saved movie IDs: {len(ids)}")
    log.info("=" * 60)
    log.info("BERT embedding build complete!")


if __name__ == "__main__":
    run()
