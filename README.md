<div align="center">

# CineScope

### AI-Powered Movie Recommendation Engine

**ALS × BERT × Knowledge Graph — Three AI models, one unified recommendation system.**

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

![Landing Page](frontend/screenshots/Landing%20page.png)

## Overview

CineScope is a hybrid movie recommendation system that combines three fundamentally different AI approaches to provide personalized movie suggestions. Unlike single-model recommenders, CineScope leverages the strengths of each model to compensate for individual weaknesses, producing richer and more diverse recommendations.

The system is built with a **FastAPI** backend serving a RESTful API, and a single-page **vanilla HTML/CSS/JavaScript** frontend with a premium cinematic design. All models are pre-trained and served at inference time, enabling sub-second response times.

| Metric | Value |
|--------|-------|
| Raw Movies Processed | **1,701,904** |
| Curated Movie Database | **~1.1 Million** |
| User Ratings | **25 Million** |
| AI Models | **3 (ALS + BERT + KG)** |
| Features Per Movie | **47** |

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Recommendation Models](#recommendation-models)
- [Hybrid Engine](#hybrid-engine)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Local Development](#local-development)
- [API Reference](#api-reference)

---

## Features

- **Netflix/Hotstar-inspired UI** — Hero carousel, horizontal scroll rows, hover-expand cards, movie detail modals
- **Three AI recommendation engines** — Collaborative filtering (ALS), semantic similarity (BERT), entity-based connections (Knowledge Graph)
- **Hybrid fusion** — Weighted combination of all three models for superior recommendations
- **Progressive loading** — Movie details load instantly, recommendations load independently with spinners
- **Configurable frontend** — Connect to any backend instance via setup screen
- **Search** — Title search with instant results
- **Genre & Decade browsing** — Filter by 18 genres and 7 decades
- **For You page** — AI-curated picks that refresh every visit
- **Explainable recommendations** — Knowledge Graph shows exactly which shared entities connect two movies
- **Real-time cast data** — Fetches live cast information from TMDB API

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Vercel)                     │
│              Static HTML / CSS / JavaScript              │
│         Netflix/Hotstar-inspired Golden Hour UI         │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Backend (SnapDeploy)                    │
│                    FastAPI Server                        │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ALS Model │  │BERT Model│  │  Knowledge Graph     │  │
│  │128 latent│  │384-dim   │  │ 2,116 entities       │  │
│  │factors   │  │embeddings│  │ sparse matrix        │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │
│       │              │                   │              │
│       └──────────────┼───────────────────┘              │
│                      ▼                                  │
│            ┌─────────────────┐                          │
│            │ Hybrid Stacker  │                          │
│            │ Weighted Fusion │                          │
│            └─────────────────┘                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              UptimeRobot (Keep Alive)                    │
│            Pings /health every 5 minutes                 │
└─────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Raw Data Sources

| Dataset | Movies | Description |
|---------|--------|-------------|
| TMDB Dataset v11 | 1,459,318 | Full cinema history — titles, genres, budgets, revenues, production companies, cast, keywords, overviews, popularity metrics |
| TMDB Supplement (2021-2025) | 232,586 | Recent releases underrepresented in base dataset |
| TMDB Top 10K (2026) | 10,000 | Top-rated movies from 2026 |
| **Total Raw** | **1,701,904** | ~700MB across three CSV files |

### Preprocessing Pipeline

| Step | Description | Impact |
|------|-------------|--------|
| **Deduplication** | Remove duplicate entries by TMDB ID | 1.7M → ~1.46M |
| **Adult Content Filter** | Filter out adult-flagged content | -9.8% |
| **Date Parsing & Year Filter** | Parse dates, extract year/month/decade, filter pre-1900 | -20% |
| **Text Cleaning** | Lowercase, strip URLs, remove special characters | -1.3% |
| **Genre Processing** | Normalize 3 formats (text, pipe-delimited, JSON IDs), map 19 TMDB genre IDs | -2.4% |
| **Vote Count Filter** | Minimum vote threshold for statistical significance | -1% |
| **Feature Engineering** | `log_budget`, `log_revenue`, `ROI`, `log_popularity`, `release_year`, `decade`, 19 one-hot genre columns | +47 features |

**Final dataset: ~1.1 million movies with 47 features**

### User Interaction Data

- **Source:** MovieLens 25M dataset
- **Ratings:** 25 million from 162,534 users across 18,465 movies
- **Split:** Temporal train/validation/test (80/10/10)

---

## Recommendation Models

### 1. ALS (Alternating Least Squares) — Collaborative Filtering

| Property | Value |
|----------|-------|
| Library | `implicit.als.AlternatingLeastSquares` |
| Input | Sparse user×movie interaction matrix |
| Dimensions | 162,534 users × 18,465 movies |
| Latent Factors | 128 |
| Iterations | 50 |
| Regularization | 0.01 |
| Confidence Scaling | 40x |
| Training Data | 20.2M interactions |

**How it works:** ALS factorizes the massive user-movie interaction matrix into two lower-dimensional matrices — user factors and item factors, each with 128 latent dimensions. The model iteratively alternates between fixing user factors and solving for item factors, then vice versa. This process discovers hidden patterns in user behavior: users who rate similar movies will have similar latent vectors, enabling the model to predict which unseen movies a user would enjoy.

**Strengths:** Captures complex behavioral patterns, discovers non-obvious taste connections.
**Weakness:** Cannot recommend movies with zero interactions (cold-start for new movies).

### 2. BERT (Bidirectional Encoder Representations) — Content Similarity

| Property | Value |
|----------|-------|
| Library | `sentence-transformers/all-MiniLM-L6-v2` |
| Input | Title + overview + tagline + keywords |
| Embedding Dim | 384 |
| Batch Size | 512 |
| Movies Encoded | 24,559 |
| Matrix Size | [24,559 × 384] (35.98 MB) |

**How it works:** BERT encodes each movie's textual description into a dense 384-dimensional vector embedding using a pre-trained transformer model fine-tuned for semantic similarity. Two movies with similar themes, tones, or narratives — even if they belong to different genres — will have embeddings that are close in the vector space.

**Strengths:** Works without user data (solves cold-start), understands semantic meaning beyond keywords.
**Weakness:** Only considers content similarity, ignores collaborative signals.

### 3. Knowledge Graph — Entity-Based Connections

| Property | Value |
|----------|-------|
| Library | `scipy.sparse` + `sklearn.metrics.pairwise.cosine_similarity` |
| Input | Genres, production companies, countries, languages |
| Matrix | 24,561 movies × 2,116 entities |
| Non-zero Entries | 142,328 |
| Sparsity | 99.7% |

**How it works:** Extracts structured entities from each movie's metadata and builds a sparse movie×entity co-occurrence matrix. Similarity is computed using cosine similarity on entity vectors. This creates **explainable recommendations** — the system can tell you exactly which shared entities connect two movies.

**Strengths:** Highly explainable, captures industry-level connections, fast inference.
**Weakness:** Limited to metadata entities, misses narrative and tonal similarity.

---

## Hybrid Engine

The Hybrid Stacker combines all three models using **weighted score fusion**:

| Model | Weight | Signal Type |
|-------|--------|-------------|
| ALS | 40% | Collaborative behavior |
| BERT | 25% | Content-based semantic similarity |
| Knowledge Graph | 20% | Entity-based connections |
| Popularity | 15% | Fallback baseline for coverage |

**Fusion process:**
1. Retrieve top-50 candidates from each available model
2. Normalize scores within each model (divide by max score)
3. Multiply normalized scores by the model's weight
4. Sum weighted scores across all models for each candidate movie
5. Re-rank by total weighted score and return top-N

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| ALS Library | implicit |
| BERT Runtime | sentence-transformers (PyTorch) |
| ML Utilities | scikit-learn |
| Data Processing | Pandas + PyArrow |
| Sparse Matrices | SciPy |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Design | Golden Hour Cinema (Netflix/Hotstar-inspired) |
| Fonts | Cormorant Garamond (headings) + DM Sans (body) |
| Backend Deployment | Docker on SnapDeploy |
| Frontend Deployment | Vercel |
| Keep-Alive | UptimeRobot |

---

## Project Structure

```
Movie recommender systems/
├── frontend/                    # Static frontend (deploy to Vercel)
│   ├── index.html               # Single-page app (45KB)
│   ├── vercel.json              # Vercel deployment config
│   └── screenshots/
│       ├── Landing page.png     # UI screenshot
│       └── About page.png       # About page screenshot
├── backend/                     # FastAPI backend (deploy via Docker)
│   ├── Dockerfile               # Python 3.12-slim, no torch (~435MB image)
│   ├── docker-compose.yml       # One-command local Docker
│   ├── requirements.txt         # Full deps (with torch, for local dev)
│   ├── requirements-prod.txt    # Prod deps (no torch, smaller image)
│   ├── run.py                   # Entry point (uvicorn)
│   ├── config/
│   │   └── settings.yaml        # Model hyperparameters, API config
│   ├── src/
│   │   ├── app.py               # FastAPI app, routes, lifespan
│   │   ├── config.py            # Config loader
│   │   ├── data/
│   │   │   ├── loader.py        # MovieData class, TMDB credits fetcher
│   │   │   ├── preprocessor.py  # Data pipeline (1.7M → ~1.1M)
│   │   │   ├── build_all.py     # Master rebuild script
│   │   │   ├── build_als.py     # ALS model trainer
│   │   │   ├── build_embeddings.py  # BERT encoder
│   │   │   └── build_kg.py      # Knowledge graph builder
│   │   └── models/
│   │       ├── als.py           # ALSRecommender class
│   │       ├── bert.py          # BERTRecommender class
│   │       ├── kg.py            # KnowledgeGraphRecommender class
│   │       └── hybrid.py        # HybridRecommender (weighted fusion)
│   ├── data/
│   │   ├── processed/           # movies_clean.parquet (16MB)
│   │   └── embeddings/          # bert_embeddings.npy (36MB)
│   └── models/
│       ├── collaborative/       # ALS model files (126MB)
│       └── graph/               # Knowledge graph (5MB)
├── .gitignore
└── README.md
```

---

## Deployment

### Step 1: Deploy Frontend → Vercel

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import your GitHub repo
3. Set **Root Directory** to `frontend`
4. Click **Deploy**
5. Your frontend is live at `https://your-project.vercel.app`

### Step 2: Deploy Backend → SnapDeploy

1. Zip the `backend/` folder
2. Go to SnapDeploy → **New Project** → Upload ZIP
3. Set environment variables:
   ```
   TMDB_API_KEY=your_api_key_here
   TMDB_READ_ACCESS_TOKEN=your_token_here
   ```
4. Deploy → Your backend is live at `https://your-app.snapdeploy.io`

### Step 3: Connect Frontend to Backend

1. Open your Vercel frontend URL
2. Enter your SnapDeploy backend URL in the setup screen
3. Click **Connect** — done!

### Step 4: Keep Alive with UptimeRobot

Your SnapDeploy backend sleeps after 30-45 minutes of inactivity. UptimeRobot prevents this:

1. Go to [uptimerobot.com](https://uptimerobot.com) → **Sign up free**
2. Click **Add New Monitor**
3. Configure:
   - **Monitor Type:** HTTP(s)
   - **Friendly Name:** CineScope API
   - **URL:** `https://your-app.snapdeploy.io/health`
   - **Monitoring Interval:** 5 minutes
4. Click **Create Monitor**

Your backend now stays awake 24/7.

---

## Local Development

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env file)
echo "TMDB_API_KEY=your_key" > .env
echo "TMDB_READ_ACCESS_TOKEN=your_token" >> .env

# Start server
python run.py
# Server runs at http://localhost:8080
```

### Frontend

Just open `frontend/index.html` in a browser. Enter `http://localhost:8080` as the backend URL when prompted.

### Docker (Local)

```bash
cd backend

# Build and run
docker compose up --build

# Or standalone
docker build -t cinescope .
docker run -p 8080:8080 -e TMDB_API_KEY=your_key cinescope
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Lightweight health check (for UptimeRobot) |
| `/` | GET | Server status with model health + movie count |
| `/api/rows` | GET | Content rows for Discover page |
| `/api/for-you` | GET | Random curated picks (3 categories) |
| `/api/search?q={query}` | GET | Title search |
| `/api/trending?n={n}` | GET | Top N trending movies |
| `/api/top-rated?n={n}` | GET | Top N highest rated movies |
| `/api/genre/{genre}?n={n}` | GET | Movies by genre |
| `/api/decade/{decade}?n={n}` | GET | Movies by decade |
| `/api/movie/{tmdb_id}` | GET | Movie details |
| `/api/movie/{tmdb_id}/credits` | GET | Cast from TMDB API |
| `/api/recommend/user/{user_id}` | GET | Hybrid recommendations by user |
| `/api/recommend/movie/{tmdb_id}` | GET | Hybrid recommendations by movie |
| `/api/recommend/movie/{tmdb_id}/all` | GET | All 3 models separately |
| `/api/similar/{tmdb_id}` | GET | Similar movies (ALS) |
| `/api/explain/{id_a}/{id_b}` | GET | Explain connection (KG) |
| `/api/stats` | GET | Dataset statistics |

---

## Screenshots

### Landing Page
![Landing Page](frontend/screenshots/Landing%20page.png)

### About Page
![About Page](frontend/screenshots/About%20page.png)

---

## License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">

**Built with ALS × BERT × Knowledge Graph**

*CineScope — AI-Powered Movie Discovery*

</div>
