"""
Data loader module with Streamlit caching.

Loads preprocessed Parquet files and provides cached access
for the dashboard. All heavy I/O happens once and is cached
across reruns.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


# Default paths relative to the project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@st.cache_data(show_spinner="Loading movie data...")
def load_movies(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load the cleaned movies Parquet file.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing processed Parquet files.
        Defaults to data/processed/ relative to project root.

    Returns
    -------
    pd.DataFrame
        Cleaned movies DataFrame with rating gaps computed.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / "movies_clean.parquet"

    if not path.exists():
        st.error(
            f"Processed data not found at `{path}`.\n\n"
            "Please run the preprocessing script first:\n\n"
            "```\npython preprocess.py\n```"
        )
        st.stop()

    df = pd.read_parquet(path)
    return df


@st.cache_data(show_spinner="Loading review data...")
def load_reviews(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load the reviews Parquet file with sentiment scores.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing processed Parquet files.

    Returns
    -------
    pd.DataFrame
        Reviews DataFrame with VADER sentiment scores.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / "reviews_sentiment.parquet"

    if not path.exists():
        st.error(
            f"Processed review data not found at `{path}`.\n\n"
            "Please run the preprocessing script first:\n\n"
            "```\npython preprocess.py\n```"
        )
        st.stop()

    df = pd.read_parquet(path)
    return df


@st.cache_data(show_spinner="Computing sentiment summaries...")
def load_movie_sentiment_summary(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load reviews and compute per-movie sentiment aggregation.

    Returns
    -------
    pd.DataFrame
        One row per movie with avg_sentiment, pct_positive, etc.
    """
    reviews = load_reviews(data_dir)

    grouped = reviews.groupby("movie_id")

    agg = grouped.agg(
        avg_sentiment=("compound", "mean"),
        median_sentiment=("compound", "median"),
        reviews_analyzed=("compound", "count"),
    ).reset_index()

    # Aggregate all sentiment labels at once instead of calling Python once per movie.
    counts = reviews.groupby(["movie_id", "sentiment_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    totals = grouped.size()
    pcts = pd.DataFrame(index=counts.index)
    for label, column in [
        ("Positive", "pct_positive"),
        ("Neutral", "pct_neutral"),
        ("Negative", "pct_negative"),
    ]:
        pcts[column] = counts.get(label, 0).div(totals).mul(100)
    pcts = pcts.reset_index()
    agg = agg.merge(pcts, on="movie_id", how="left")

    return agg


@st.cache_data(show_spinner="Preparing dashboard data...")
def load_merged_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load movies and merge with sentiment summaries.

    Returns
    -------
    pd.DataFrame
        Movies DataFrame enriched with per-movie sentiment columns.
    """
    movies = load_movies(data_dir)
    sentiment = load_movie_sentiment_summary(data_dir)

    merged = movies.merge(sentiment, on="movie_id", how="left")
    return merged


def get_all_genres(movies_df: pd.DataFrame) -> list[str]:
    """Extract all unique genres from the dataset, sorted."""
    all_genres = set()
    for genre_list in movies_df["genres_list"]:
        if isinstance(genre_list, list):
            all_genres.update(genre_list)
    return sorted(g for g in all_genres if g)


def get_year_range(movies_df: pd.DataFrame) -> tuple[int, int]:
    """Get the min and max release years in the dataset."""
    valid_years = movies_df["release_year"].dropna()
    if len(valid_years) == 0:
        return 1900, 2030
    return int(valid_years.min()), int(valid_years.max())
