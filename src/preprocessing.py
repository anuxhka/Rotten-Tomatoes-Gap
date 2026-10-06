"""
Data preprocessing module for the Rotten Tomatoes Gap dashboard.

Cleans, validates, and transforms raw CSV data from the Kaggle dataset
into analysis-ready DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# --- Column mappings from raw dataset to internal names ---

MOVIES_COLUMN_MAP = {
    "rotten_tomatoes_link": "movie_id",
    "movie_title": "title",
    "original_release_date": "release_date",
    "genres": "genres",
    "tomatometer_rating": "critic_rating",
    "audience_rating": "audience_rating",
    "tomatometer_count": "critic_count",
    "audience_count": "audience_count",
    "content_rating": "content_rating",
    "movie_info": "synopsis",
    "directors": "directors",
    "runtime": "runtime",
    "tomatometer_status": "tomatometer_status",
    "tomatometer_top_critics_count": "top_critics_count",
    "tomatometer_fresh_critics_count": "fresh_critics_count",
    "tomatometer_rotten_critics_count": "rotten_critics_count",
}

REVIEWS_COLUMN_MAP = {
    "rotten_tomatoes_link": "movie_id",
    "critic_name": "critic_name",
    "publisher_name": "publication",
    "review_date": "review_date",
    "review_type": "review_type",
    "review_content": "review_text",
    "top_critic": "top_critic",
    "review_score": "review_score",
}


def clean_movies(raw_movies_path: str | Path) -> pd.DataFrame:
    """
    Load and clean the movies CSV file.

    Steps:
    1. Read CSV and rename columns to internal names.
    2. Convert rating columns to numeric (NaN on failure, never zero).
    3. Validate ratings are in [0, 100] range.
    4. Parse release dates and derive release year.
    5. Compute rating gap and absolute gap.
    6. Split genres into lists.

    Parameters
    ----------
    raw_movies_path : str or Path
        Path to the raw rotten_tomatoes_movies.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned movies DataFrame.
    """
    print(f"  Loading movies from {raw_movies_path}...")
    df = pd.read_csv(raw_movies_path)
    print(f"  Raw movies loaded: {len(df)} records, {len(df.columns)} columns")

    # Rename columns that exist in the file
    rename_map = {k: v for k, v in MOVIES_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Report any expected columns that are missing
    expected_internal = set(MOVIES_COLUMN_MAP.values())
    actual_internal = set(rename_map.values())
    missing = expected_internal - actual_internal
    if missing:
        print(f"  Warning: Missing expected columns: {missing}")

    # Convert ratings to numeric (coerce errors → NaN, NOT zero)
    for col in ["critic_rating", "audience_rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert counts to numeric
    for col in ["critic_count", "audience_count", "top_critics_count",
                "fresh_critics_count", "rotten_critics_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate rating percentages: must be in [0, 100], else NaN
    for col in ["critic_rating", "audience_rating"]:
        if col in df.columns:
            mask = (df[col] < 0) | (df[col] > 100)
            invalid_count = mask.sum()
            if invalid_count > 0:
                print(f"  Warning: {invalid_count} out-of-range values in {col}, setting to NaN")
                df.loc[mask, col] = np.nan

    # Parse release date and derive year
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year.astype("Int64")
    else:
        df["release_year"] = pd.NA

    # Compute rating gap: critic - audience
    if "critic_rating" in df.columns and "audience_rating" in df.columns:
        df["rating_gap"] = df["critic_rating"] - df["audience_rating"]
        df["abs_gap"] = df["rating_gap"].abs()
        df["has_both_ratings"] = df["critic_rating"].notna() & df["audience_rating"].notna()
    else:
        df["rating_gap"] = np.nan
        df["abs_gap"] = np.nan
        df["has_both_ratings"] = False

    # Split genres into lists (for filtering)
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].fillna("").str.split(r",\s*")
        # Clean up empty strings in lists
        df["genres_list"] = df["genres_list"].apply(
            lambda x: [g.strip() for g in x if g.strip()] if isinstance(x, list) else []
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    # Extract runtime as numeric minutes
    if "runtime" in df.columns:
        df["runtime_minutes"] = pd.to_numeric(
            df["runtime"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        ).astype("Int64")

    # Ensure movie_id is string
    if "movie_id" in df.columns:
        df["movie_id"] = df["movie_id"].astype(str)

    # Create display label: "Title (Year)"
    df["display_label"] = df.apply(
        lambda row: f"{row.get('title', 'Unknown')} ({int(row['release_year'])})"
        if pd.notna(row.get("release_year"))
        else str(row.get("title", "Unknown")),
        axis=1,
    )

    print(f"  Movies cleaned: {len(df)} records")
    print(f"    - With both ratings: {df['has_both_ratings'].sum()}")
    print(f"    - Missing critic rating: {df['critic_rating'].isna().sum()}")
    print(f"    - Missing audience rating: {df['audience_rating'].isna().sum()}")

    return df


def clean_reviews(raw_reviews_path: str | Path) -> pd.DataFrame:
    """
    Load and clean the critic reviews CSV file.

    Steps:
    1. Read CSV and rename columns.
    2. Remove records with empty/null review text.
    3. Remove exact duplicate rows.
    4. Parse review dates.
    5. Ensure movie_id is string for joining.

    Parameters
    ----------
    raw_reviews_path : str or Path
        Path to the raw rotten_tomatoes_critic_reviews.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned reviews DataFrame (without sentiment scores yet).
    """
    print(f"  Loading critic reviews from {raw_reviews_path}...")
    df = pd.read_csv(raw_reviews_path)
    print(f"  Raw reviews loaded: {len(df)} records, {len(df.columns)} columns")

    # Rename columns that exist
    rename_map = {k: v for k, v in REVIEWS_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure movie_id is string
    if "movie_id" in df.columns:
        df["movie_id"] = df["movie_id"].astype(str)

    # Remove records with empty/null review text
    before = len(df)
    if "review_text" in df.columns:
        df = df[df["review_text"].notna() & (df["review_text"].astype(str).str.strip() != "")]
    removed_empty = before - len(df)
    print(f"  Removed {removed_empty} reviews with empty text")

    # Remove exact duplicate rows (same movie_id + critic_name + review_text)
    # but preserve distinct reviews even if text is similar
    before = len(df)
    dedup_cols = ["movie_id", "critic_name", "review_text"]
    existing_dedup = [c for c in dedup_cols if c in df.columns]
    if existing_dedup:
        df = df.drop_duplicates(subset=existing_dedup, keep="first")
    removed_dupes = before - len(df)
    print(f"  Removed {removed_dupes} exact duplicate reviews")

    # Parse review date
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    print(f"  Reviews cleaned: {len(df)} records")

    return df


def validate_join(movies_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict:
    """
    Validate the join between movies and reviews on movie_id.

    Returns
    -------
    dict
        Statistics about the join quality.
    """
    movie_ids = set(movies_df["movie_id"].unique())
    review_movie_ids = set(reviews_df["movie_id"].unique())

    stats = {
        "total_movies": len(movie_ids),
        "total_review_movie_ids": len(review_movie_ids),
        "movies_with_reviews": len(movie_ids & review_movie_ids),
        "movies_without_reviews": len(movie_ids - review_movie_ids),
        "reviews_without_movies": len(review_movie_ids - movie_ids),
    }

    print(f"  Join validation:")
    print(f"    - Total movies: {stats['total_movies']}")
    print(f"    - Movies with reviews: {stats['movies_with_reviews']}")
    print(f"    - Movies without reviews: {stats['movies_without_reviews']}")
    print(f"    - Review movie_ids not in movies table: {stats['reviews_without_movies']}")

    return stats
