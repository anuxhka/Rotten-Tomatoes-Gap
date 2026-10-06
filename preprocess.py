"""
One-time preprocessing script for the Rotten Tomatoes Gap dashboard.

Downloads (or locates) the raw dataset, cleans it, runs VADER sentiment
analysis on all critic reviews, and saves processed Parquet files.

Usage:
    python preprocess.py

The script will:
1. Attempt to download the dataset via KaggleHub, or locate it in data/raw/
2. Clean and validate the movies data
3. Clean the critic reviews data
4. Score all reviews with VADER sentiment analysis
5. Save processed files to data/processed/
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def find_dataset() -> tuple[Path, Path]:
    """
    Locate the raw dataset files.

    Tries KaggleHub first, then falls back to data/raw/ directory.

    Returns
    -------
    tuple[Path, Path]
        (movies_path, reviews_path)
    """
    movies_name = "rotten_tomatoes_movies.csv"
    reviews_name = "rotten_tomatoes_critic_reviews.csv"

    # Try KaggleHub download first
    print("Attempting to locate dataset via KaggleHub...")
    try:
        import kagglehub
        dataset_path = Path(kagglehub.dataset_download(
            "stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset"
        ))
        movies_path = dataset_path / movies_name
        reviews_path = dataset_path / reviews_name
        if movies_path.exists() and reviews_path.exists():
            print(f"  Dataset found via KaggleHub: {dataset_path}")
            return movies_path, reviews_path
        else:
            print(f"  KaggleHub path exists but files not found at expected names.")
    except Exception as e:
        print(f"  KaggleHub download failed: {e}")
        print(f"  Falling back to local data/raw/ directory...")

    # Fallback to local data/raw/
    raw_dir = Path(__file__).parent / "data" / "raw"
    movies_path = raw_dir / movies_name
    reviews_path = raw_dir / reviews_name

    if movies_path.exists() and reviews_path.exists():
        print(f"  Dataset found in local directory: {raw_dir}")
        return movies_path, reviews_path

    # Neither source available
    print(f"\n  ERROR: Could not find dataset files.")
    print(f"  Looked for:")
    print(f"    - KaggleHub cache")
    print(f"    - {raw_dir / movies_name}")
    print(f"    - {raw_dir / reviews_name}")
    print(f"\n  Please download the dataset from:")
    print(f"  https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset")
    print(f"  and place the CSV files in: {raw_dir}")
    sys.exit(1)


def main():
    """Run the full preprocessing pipeline."""
    start_time = time.time()
    print("=" * 60)
    print("Rotten Tomatoes Gap — Data Preprocessing Pipeline")
    print("=" * 60)

    # --- Step 1: Locate dataset ---
    print("\n[1/5] Locating dataset files...")
    movies_path, reviews_path = find_dataset()

    # --- Step 2: Clean movies ---
    print("\n[2/5] Cleaning movies data...")
    from src.preprocessing import clean_movies
    movies_df = clean_movies(movies_path)

    # --- Step 3: Clean reviews ---
    print("\n[3/5] Cleaning critic reviews...")
    from src.preprocessing import clean_reviews, validate_join
    reviews_df = clean_reviews(reviews_path)

    # Validate join
    print("\n  Validating movie-review join...")
    join_stats = validate_join(movies_df, reviews_df)

    # --- Step 4: VADER sentiment scoring ---
    print("\n[4/5] Running VADER sentiment analysis on critic reviews...")
    from src.sentiment import score_reviews
    reviews_scored = score_reviews(reviews_df)

    # --- Step 5: Save processed data ---
    print("\n[5/5] Saving processed data...")
    output_dir = Path(__file__).parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    movies_out = output_dir / "movies_clean.parquet"
    reviews_out = output_dir / "reviews_sentiment.parquet"

    movies_df.to_parquet(movies_out, index=False)
    print(f"  Saved: {movies_out} ({movies_out.stat().st_size / 1e6:.1f} MB)")

    reviews_scored.to_parquet(reviews_out, index=False)
    print(f"  Saved: {reviews_out} ({reviews_out.stat().st_size / 1e6:.1f} MB)")

    # --- Summary ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Movies: {len(movies_df):,} records")
    print(f"    - With both ratings: {movies_df['has_both_ratings'].sum():,}")
    print(f"    - Mean absolute gap: {movies_df.loc[movies_df['has_both_ratings'], 'abs_gap'].mean():.1f} pp")
    print(f"  Reviews: {len(reviews_scored):,} records")
    print(f"    - Covering {reviews_scored['movie_id'].nunique():,} movies")
    print(f"\nDashboard data ready. Launch with:")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
