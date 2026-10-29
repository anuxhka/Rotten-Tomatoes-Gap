import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.data_loader import get_all_genres, get_year_range, load_movie_sentiment_summary


class DataLoaderTests(unittest.TestCase):
    def test_empty_review_snapshot_has_empty_summary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            pd.DataFrame({
                "movie_id": pd.Series(dtype="str"),
                "compound": pd.Series(dtype="float64"),
                "sentiment_label": pd.Series(dtype="str"),
            }).to_parquet(Path(directory) / "reviews_sentiment.parquet")
            summary = load_movie_sentiment_summary(directory)
        self.assertTrue(summary.empty)
        self.assertIn("pct_positive", summary.columns)

    def test_genre_and_year_helpers_handle_missing_values(self):
        movies = pd.DataFrame({
            "genres_list": [["Comedy", "Drama"], ["Drama"], []],
            "release_year": pd.Series([1990, None, 2020], dtype="Int64"),
        })
        self.assertEqual(get_all_genres(movies), ["Comedy", "Drama"])
        self.assertEqual(get_year_range(movies), (1990, 2020))


if __name__ == "__main__":
    unittest.main()
