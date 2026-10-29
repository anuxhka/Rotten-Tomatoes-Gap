"""
Sentiment analysis module for critic review text.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to score
critic review text. Produces per-review scores and per-movie aggregations.

Important: This analyzes CRITIC review text only. Audience text reviews
are not available in this dataset.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# VADER sentiment thresholds (standard)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def classify_sentiment(compound: float) -> str:
    """
    Classify a VADER compound score into a sentiment label.

    Parameters
    ----------
    compound : float
        VADER compound score in [-1, 1].

    Returns
    -------
    str
        'Positive', 'Negative', or 'Neutral'.
    """
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def score_reviews(reviews_df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to each review.

    Preserves original text (no lowercasing, no punctuation removal).
    VADER handles raw text well, including punctuation, capitalization,
    and negation.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        DataFrame with a text column to analyze.
    text_column : str
        Name of the column containing review text.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: compound, pos, neu, neg, sentiment_label.
    """
    df = reviews_df.copy()

    print(f"  Scoring {len(df)} reviews with VADER...")

    if df.empty:
        for column in ["compound", "vader_pos", "vader_neu", "vader_neg"]:
            df[column] = pd.Series(dtype="float64")
        df["sentiment_label"] = pd.Series(dtype="str")
        print("  No reviews to score")
        return df

    # Score each review — preserve original text
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_column].apply(lambda text: analyzer.polarity_scores(str(text)))

    df["compound"] = scores.apply(lambda x: x["compound"])
    df["vader_pos"] = scores.apply(lambda x: x["pos"])
    df["vader_neu"] = scores.apply(lambda x: x["neu"])
    df["vader_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_label"] = df["compound"].apply(classify_sentiment)

    # Summary stats
    label_counts = df["sentiment_label"].value_counts()
    print(f"  Sentiment scoring complete:")
    for label in ["Positive", "Neutral", "Negative"]:
        count = label_counts.get(label, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"    - {label}: {count} ({pct:.1f}%)")

    return df


def aggregate_movie_sentiment(scored_reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-review sentiment scores to per-movie summaries.

    Parameters
    ----------
    scored_reviews_df : pd.DataFrame
        Reviews DataFrame with VADER scores and sentiment_label.

    Returns
    -------
    pd.DataFrame
        One row per movie with sentiment summary columns.
    """
    print("  Aggregating movie-level sentiment...")

    if scored_reviews_df.empty:
        return pd.DataFrame({
            "movie_id": pd.Series(dtype="str"),
            "avg_sentiment": pd.Series(dtype="float64"),
            "median_sentiment": pd.Series(dtype="float64"),
            "std_sentiment": pd.Series(dtype="float64"),
            "reviews_analyzed": pd.Series(dtype="int64"),
            "pct_positive": pd.Series(dtype="float64"),
            "pct_neutral": pd.Series(dtype="float64"),
            "pct_negative": pd.Series(dtype="float64"),
            "movie_sentiment_label": pd.Series(dtype="str"),
        })

    grouped = scored_reviews_df.groupby("movie_id")

    agg = grouped.agg(
        avg_sentiment=("compound", "mean"),
        median_sentiment=("compound", "median"),
        std_sentiment=("compound", "std"),
        reviews_analyzed=("compound", "count"),
    ).reset_index()

    # Calculate percentage of each sentiment class per movie
    def sentiment_percentages(group):
        total = len(group)
        if total == 0:
            return pd.Series({
                "pct_positive": 0.0,
                "pct_neutral": 0.0,
                "pct_negative": 0.0,
            })
        counts = group["sentiment_label"].value_counts()
        return pd.Series({
            "pct_positive": counts.get("Positive", 0) / total * 100,
            "pct_neutral": counts.get("Neutral", 0) / total * 100,
            "pct_negative": counts.get("Negative", 0) / total * 100,
        })

    pcts = grouped.apply(sentiment_percentages, include_groups=False).reset_index()
    agg = agg.merge(pcts, on="movie_id", how="left")

    # Classify overall movie sentiment
    agg["movie_sentiment_label"] = agg["avg_sentiment"].apply(classify_sentiment)

    print(f"  Movie sentiment aggregated for {len(agg)} movies")

    return agg
