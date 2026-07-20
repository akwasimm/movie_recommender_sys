"""
CineScope Data Preprocessor
Merges 3 raw CSV datasets → cleans → engineers features → outputs movies_clean.parquet
"""
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("preprocessor")

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "csv data unprocessed"
OUT_DIR = ROOT / "data" / "processed"

GENRE_MAP = {
    28: "action", 12: "adventure", 16: "animation", 35: "comedy",
    80: "crime", 99: "documentary", 18: "drama", 10751: "family",
    14: "fantasy", 36: "history", 27: "horror", 10402: "music",
    9648: "mystery", 10749: "romance", 878: "sci-fi", 10770: "tv-movie",
    53: "thriller", 10752: "war", 37: "western",
}

KEEP_GENRES = {
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "sci-fi", "thriller", "war", "western",
}

MIN_VOTE_COUNT = 50
MIN_RELEASE_YEAR = 1900


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z0-9\s\-\']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_genres(raw):
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if s.startswith("["):
        try:
            import ast
            ids = ast.literal_eval(s)
            return [GENRE_MAP.get(i, "") for i in ids if i in GENRE_MAP]
        except Exception:
            pass
    parts = re.split(r"[|,;/]", s)
    return [clean_text(p) for p in parts if p.strip()]


def load_base():
    path = RAW_DIR / "TMDB_movie_dataset_v11.csv"
    log.info(f"Loading base dataset: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"id": "tmdb_id"})
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    log.info(f"  Base: {len(df)} rows")
    return df


def load_supplement():
    path = RAW_DIR / "tmdb_movies_2021_2025.csv"
    log.info(f"Loading supplement: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"tmdb_id": "tmdb_id", "poster_url": "poster_path"})
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    if "poster_path" in df.columns:
        df["poster_path"] = df["poster_path"].apply(
            lambda x: str(x).replace("https://image.tmdb.org/t/p/w500", "")
            if pd.notna(x) and str(x).startswith("http") else x
        )
    log.info(f"  Supplement: {len(df)} rows")
    return df


def load_gap():
    path = RAW_DIR / "tmdb_top_10k_movies_2026.csv.csv"
    log.info(f"Loading gap fill: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={"id": "tmdb_id"})
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    if "genre_ids" in df.columns:
        df["genres"] = df["genre_ids"]
    log.info(f"  Gap fill: {len(df)} rows")
    return df


def merge_datasets():
    base = load_base()
    supp = load_supplement()
    gap = load_gap()

    # Align schemas: pick a canonical set of columns
    canonical = [
        "tmdb_id", "title", "original_title", "release_date", "vote_average",
        "vote_count", "popularity", "original_language", "overview",
        "poster_path", "backdrop_path", "runtime", "budget", "revenue",
        "tagline", "keywords", "production_companies", "production_countries",
        "spoken_languages", "imdb_id", "status", "adult", "genres",
    ]

    for df in [base, supp, gap]:
        for c in canonical:
            if c not in df.columns:
                df[c] = np.nan

    base = base[canonical]
    supp = supp[canonical]
    gap = gap[canonical]

    merged = pd.concat([base, supp, gap], ignore_index=True)
    log.info(f"  Merged: {len(merged)} rows")

    merged = merged.drop_duplicates(subset=["tmdb_id"], keep="first")
    log.info(f"  After dedup: {len(merged)} rows")

    return merged


def filter_adult(df):
    before = len(df)
    if "adult" in df.columns:
        mask = df["adult"].astype(str).str.lower().isin(["false", "0", "nan", ""])
        df = df[mask | df["adult"].isna()]
    log.info(f"  Adult filter: {before} → {len(df)}")
    return df


def parse_dates(df):
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["decade"] = (df["release_year"] // 10 * 10).astype("Int64")
    mask = df["release_year"] >= MIN_RELEASE_YEAR
    log.info(f"  Date filter (>= {MIN_RELEASE_YEAR}): {mask.sum()} valid")
    df = df[mask | df["release_year"].isna()]
    return df


def process_genres(df):
    df["genre_list"] = df["genres"].apply(normalize_genres)
    df["genres_clean"] = df["genre_list"].apply(lambda x: "|".join(sorted(x)) if x else "")

    df = df[df["genre_list"].apply(lambda x: len(x) > 0 and all(g in KEEP_GENRES for g in x))]
    log.info(f"  After genre filter: {len(df)} rows")

    for g in sorted(KEEP_GENRES):
        df[f"genre_{g}"] = df["genre_list"].apply(lambda x: 1 if g in x else 0).astype("int8")

    return df


def filter_vote_count(df):
    before = len(df)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df = df[df["vote_count"] >= MIN_VOTE_COUNT]
    log.info(f"  Vote count >= {MIN_VOTE_COUNT}: {before} → {len(df)}")
    return df


def clean_text_fields(df):
    for col in ["overview", "tagline", "keywords"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["text_features"] = (
        df["title"].fillna("").apply(clean_text) + " " +
        df["overview"].apply(clean_text) + " " +
        df["tagline"].apply(clean_text) + " " +
        df["keywords"].apply(clean_text)
    )
    df["text_features"] = df["text_features"].str.strip()
    return df


def engineer_features(df):
    for col in ["budget", "revenue", "popularity"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])
    df["roi"] = np.where(df["budget"] > 0, df["revenue"] / df["budget"], 0)
    df["log_popularity"] = np.log1p(df["popularity"])

    for col in ["runtime", "vote_average"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["production_companies", "production_countries", "spoken_languages", "imdb_id", "status"]:
        if col not in df.columns:
            df[col] = ""

    return df


def run():
    log.info("=" * 60)
    log.info("CineScope Data Pipeline - Starting")
    log.info("=" * 60)

    merged = merge_datasets()
    merged = filter_adult(merged)
    merged = parse_dates(merged)
    merged = process_genres(merged)
    merged = filter_vote_count(merged)
    merged = clean_text_fields(merged)
    merged = engineer_features(merged)

    keep_cols = [
        "tmdb_id", "title", "original_title", "release_date", "release_year",
        "release_month", "decade", "genres_clean", "vote_average", "vote_count",
        "popularity", "log_popularity", "overview", "tagline", "keywords",
        "text_features", "original_language", "runtime", "poster_path",
        "backdrop_path", "log_budget", "log_revenue", "roi",
        "production_companies", "production_countries", "spoken_languages",
        "imdb_id", "status",
    ]
    keep_cols += [c for c in sorted(merged.columns) if c.startswith("genre_")]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    keep_cols = list(dict.fromkeys(keep_cols))
    merged = merged[keep_cols]

    merged = merged.sort_values("popularity", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "movies_clean.parquet"
    merged.to_parquet(out_path, index=False)
    log.info(f"Saved: {out_path} ({len(merged)} rows, {len(merged.columns)} cols)")
    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)

    return merged


if __name__ == "__main__":
    run()
