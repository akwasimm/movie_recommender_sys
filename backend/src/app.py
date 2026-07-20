import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path

from src.config import get_config, env
from src.data.loader import MovieData, fetch_credits
from src.models.als import ALSRecommender
from src.models.bert import BERTRecommender
from src.models.kg import KnowledgeGraphRecommender
from src.models.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cinescope")

cfg = get_config()
API_KEY = env("TMDB_API_KEY")

movie_data = MovieData()
als_model = ALSRecommender()
bert_model = BERTRecommender()
kg_model = KnowledgeGraphRecommender()
hybrid_model: HybridRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hybrid_model
    log.info("Loading CineScope models...")
    data_ok = movie_data.load()
    if data_ok:
        log.info(f"Movies loaded: {len(movie_data.movies)} titles")
    else:
        log.warning("Could not load movie data")
    als_ok = als_model.load()
    bert_ok = bert_model.load()
    kg_ok = kg_model.load()
    log.info(f"Models loaded - ALS:{als_ok} BERT:{bert_ok} KG:{kg_ok}")
    hybrid_model = HybridRecommender(
        als=als_model if als_ok else None,
        bert=bert_model if bert_ok else None,
        kg=kg_model if kg_ok else None,
        movies_df=movie_data.movies,
    )
    yield
    log.info("CineScope shutting down")


app = FastAPI(title="CineScope", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    """Lightweight health check for UptimeRobot / load balancers."""
    return JSONResponse(content={"status": "ok"})


@app.get("/")
def root():
    return {
        "name": "CineScope", "version": "1.0.0", "status": "ok",
        "models": {
            "als": als_model.model is not None,
            "bert": bert_model.embeddings is not None,
            "kg": kg_model.entity_movie_matrix is not None,
        },
        "total_movies": len(movie_data.movies) if movie_data.movies is not None else 0,
    }


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    return movie_data.search(q, limit)


@app.get("/api/movie/{tmdb_id}")
def get_movie(tmdb_id: int):
    m = movie_data.get_movie(tmdb_id)
    if not m:
        raise HTTPException(404, "Movie not found")
    return m


@app.get("/api/movie/{tmdb_id}/credits")
def get_credits(tmdb_id: int):
    return fetch_credits(tmdb_id, API_KEY)


@app.get("/api/trending")
def trending(n: int = Query(20, ge=1, le=50)):
    return movie_data.trending(n)


@app.get("/api/top-rated")
def top_rated(n: int = Query(20, ge=1, le=50)):
    return movie_data.top_rated(n)


@app.get("/api/genre/{genre}")
def by_genre(genre: str, n: int = Query(20, ge=1, le=50)):
    return movie_data.by_genre(genre, n)


@app.get("/api/decade/{decade}")
def by_decade(decade: int, n: int = Query(20, ge=1, le=50)):
    return movie_data.by_decade(decade, n)


@app.get("/api/rows")
def content_rows():
    rows = [
        {"title": "Trending Now", "movies": movie_data.trending(20)},
        {"title": "Top Rated", "movies": movie_data.top_rated(20)},
    ]
    for genre in ["action", "comedy", "drama", "sci-fi", "horror", "thriller", "romance", "animation"]:
        movies = movie_data.by_genre(genre, 20)
        if movies:
            rows.append({"title": f"Best {genre.title()} Films", "movies": movies})
    for decade in [2020, 2010, 2000, 1990]:
        movies = movie_data.by_decade(decade, 20)
        if movies:
            rows.append({"title": f"{decade}s Movies", "movies": movies})
    return rows


@app.get("/api/for-you")
def for_you():
    import random
    categories = [
        ("Hidden Gems Worth Discovering", lambda: movie_data.random_top(15, min_votes=100)),
        ("Feel-Good Comedies", lambda: movie_data.random_by_genre("comedy", 15)),
        ("Edge-of-Your-Seat Thrillers", lambda: movie_data.random_by_genre("thriller", 15)),
        ("Critically Acclaimed Dramas", lambda: movie_data.random_by_genre("drama", 15)),
        ("Sci-Fi Adventures", lambda: movie_data.random_by_genre("sci-fi", 15)),
        ("Romantic Classics", lambda: movie_data.random_by_genre("romance", 15)),
        ("Horror Night Picks", lambda: movie_data.random_by_genre("horror", 15)),
        ("Action Blockbusters", lambda: movie_data.random_by_genre("action", 15)),
        ("90s Nostalgia", lambda: movie_data.random_by_decade(1990, 15)),
        ("2000s Favorites", lambda: movie_data.random_by_decade(2000, 15)),
        ("2020s New Releases", lambda: movie_data.random_by_decade(2020, 15)),
        ("Animation for Everyone", lambda: movie_data.random_by_genre("animation", 15)),
    ]
    chosen = random.sample(categories, k=min(3, len(categories)))
    return [{"title": title, "movies": fn()} for title, fn in chosen]


@app.get("/api/recommend/user/{user_id}")
def recommend_user(user_id: int, n: int = Query(10, ge=1, le=30)):
    recs = hybrid_model.recommend(user_id=user_id, n=n)
    return _enrich(recs)


@app.get("/api/recommend/movie/{tmdb_id}")
def recommend_movie(tmdb_id: int, n: int = Query(10, ge=1, le=30)):
    recs = hybrid_model.recommend(tmdb_id=tmdb_id, n=n)
    return _enrich(recs)


@app.get("/api/recommend/movie/{tmdb_id}/all")
def all_recommendations(tmdb_id: int, n: int = Query(6, ge=1, le=20)):
    return {
        "bert": _enrich(bert_model.recommend_from_movie(tmdb_id, n)),
        "als": _enrich(als_model.similar_items(tmdb_id, n)),
        "kg": _enrich(kg_model.recommend(tmdb_id, n)),
    }


@app.get("/api/similar/{tmdb_id}")
def similar(tmdb_id: int, n: int = Query(10, ge=1, le=30)):
    recs = als_model.similar_items(tmdb_id, n)
    return _enrich(recs)


@app.get("/api/explain/{id_a}/{id_b}")
def explain(id_a: int, id_b: int):
    return kg_model.explain(id_a, id_b)


@app.get("/api/stats")
def stats():
    return {
        "total_movies": len(movie_data.movies) if movie_data.movies is not None else 0,
        "als_users": len(als_model.user_map) if als_model.user_map else 0,
        "als_movies": len(als_model.movie_map) if als_model.movie_map else 0,
        "bert_movies": len(bert_model.movie_ids) if bert_model.movie_ids is not None else 0,
        "kg_movies": len(kg_model.movie_id_array) if kg_model.movie_id_array is not None else 0,
    }


def _enrich(recs: list[dict]) -> list[dict]:
    out = []
    for r in recs:
        m = movie_data.get_movie(r["tmdb_id"])
        if m:
            out.append({**m, "score": r["score"]})
    return out
