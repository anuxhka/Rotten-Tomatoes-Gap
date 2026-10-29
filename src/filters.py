"""Movie search and genre matching shared by the dashboard."""

import re

import pandas as pd


def matching_titles(titles: pd.Series, query: str, match_case: bool = False,
                    whole_word: bool = False) -> pd.Series:
    """Match literal title text, optionally respecting case and word boundaries."""
    query = query.strip()
    if not query:
        return pd.Series(True, index=titles.index)
    pattern = re.escape(query)
    if whole_word:
        pattern = rf"(?<!\w){pattern}(?!\w)"
    return titles.fillna("").str.contains(
        pattern, case=match_case, regex=True, na=False
    )


def genre_values(value) -> list[str]:
    """Accept list-like genres from preprocessing or a Parquet round trip."""
    values = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(values, (list, tuple)):
        return []
    return [str(genre).strip() for genre in values if str(genre).strip()]
