"""
CineScope Full Rebuild Script
Runs all data pipelines in sequence: preprocess → BERT → KG
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("  CineScope Full Data Rebuild")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/3] Preprocessing raw data...")
    from src.data.preprocessor import run as preprocess
    preprocess()

    print("\n[2/3] Building BERT embeddings...")
    from src.data.build_embeddings import run as build_bert
    build_bert()

    print("\n[3/3] Building Knowledge Graph...")
    from src.data.build_kg import run as build_kg
    build_kg()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Full rebuild complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
