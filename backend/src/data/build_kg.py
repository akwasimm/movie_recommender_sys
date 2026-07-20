"""
CineScope Knowledge Graph Builder
Extracts entities from movie metadata, builds sparse co-occurrence matrix
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kg_builder")

ROOT = Path(__file__).resolve().parent.parent.parent
MIN_ENTITY_FREQ = 5


def extract_entities(row):
    entities = set()
    genres = row.get("genres_clean", "")
    if genres:
        for g in str(genres).split("|"):
            if g.strip():
                entities.add(f"genre:{g.strip()}")

    for field, prefix, limit in [
        ("production_companies", "company", 3),
        ("production_countries", "country", 3),
        ("original_language", "lang", 1),
    ]:
        val = row.get(field, "")
        if pd.isna(val) or not val:
            continue
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
        for p in parts[:limit]:
            entities.add(f"{prefix}:{p.lower()}")

    return entities


def run():
    log.info("=" * 60)
    log.info("CineScope Knowledge Graph Builder")
    log.info("=" * 60)

    parquet_path = ROOT / "data" / "processed" / "movies_clean.parquet"
    if not parquet_path.exists():
        log.error(f"Run preprocessor first: {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)
    log.info(f"Loaded {len(df)} movies")

    log.info("Extracting entities from metadata...")
    movie_entities = {}
    entity_freq = {}

    for idx, row in df.iterrows():
        tid = int(row["tmdb_id"])
        ents = extract_entities(row)
        if ents:
            movie_entities[tid] = ents
            for e in ents:
                entity_freq[e] = entity_freq.get(e, 0) + 1

        if (idx + 1) % 100000 == 0:
            log.info(f"  Processed {idx + 1}/{len(df)} movies")

    log.info(f"Movies with entities: {len(movie_entities)}")
    log.info(f"Total unique entities: {len(entity_freq)}")

    valid_entities = {e for e, c in entity_freq.items() if c >= MIN_ENTITY_FREQ}
    log.info(f"Entities with freq >= {MIN_ENTITY_FREQ}: {len(valid_entities)}")

    entity_names = sorted(valid_entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_names)}

    movie_id_array = np.array(sorted(movie_entities.keys()))
    movie_to_idx = {int(mid): i for i, mid in enumerate(movie_id_array)}

    log.info("Building sparse entity matrix...")
    rows, cols, data = [], [], []
    for tid, ents in movie_entities.items():
        midx = movie_to_idx[tid]
        for e in ents:
            if e in entity_to_idx:
                rows.append(midx)
                cols.append(entity_to_idx[e])
                data.append(1.0)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(movie_id_array), len(entity_names)),
    )
    log.info(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")

    out_path = ROOT / "models" / "graph" / "knowledge_graph.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "entity_names": entity_names,
            "movie_id_array": movie_id_array,
            "entity_movie_matrix": matrix,
            "movie_entities": movie_entities,
        }, f)

    log.info(f"Saved: {out_path}")
    log.info("=" * 60)
    log.info("Knowledge Graph build complete!")


if __name__ == "__main__":
    run()
