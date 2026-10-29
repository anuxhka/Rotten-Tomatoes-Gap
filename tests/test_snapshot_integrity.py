"""Check the bundled snapshot against the values the dashboard displays."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


class SnapshotIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies = pd.read_parquet(DATA_DIR / "movies_clean.parquet")
        cls.reviews = pd.read_parquet(DATA_DIR / "reviews_sentiment.parquet")

    def test_movie_keys_ratings_and_derived_gaps(self):
        movies = self.movies
        self.assertFalse(movies.empty)
        self.assertFalse(movies["movie_id"].isna().any())
        self.assertFalse(movies["movie_id"].duplicated().any())
        self.assertFalse(movies["title"].isna().any())
        self.assertFalse(movies["title"].astype(str).str.strip().eq("").any())
        for column in ["critic_rating", "audience_rating"]:
            self.assertTrue(movies[column].dropna().between(0, 100).all())
        for column in ["critic_count", "audience_count"]:
            self.assertTrue(movies[column].dropna().ge(0).all())
        both = movies["critic_rating"].notna() & movies["audience_rating"].notna()
        self.assertTrue(movies["has_both_ratings"].eq(both).all())
        np.testing.assert_allclose(
            movies.loc[both, "rating_gap"],
            movies.loc[both, "critic_rating"] - movies.loc[both, "audience_rating"],
        )
        np.testing.assert_allclose(
            movies.loc[both, "abs_gap"], movies.loc[both, "rating_gap"].abs(),
        )

    def test_review_keys_scores_and_labels(self):
        reviews = self.reviews
        self.assertFalse(reviews.empty)
        for column in ["movie_id", "review_text", "compound", "sentiment_label"]:
            self.assertFalse(reviews[column].isna().any(), column)
        self.assertFalse(reviews["review_text"].astype(str).str.strip().eq("").any())
        self.assertTrue(reviews["compound"].between(-1, 1).all())
        self.assertTrue(reviews["review_type"].isin(["Fresh", "Rotten"]).all())
        expected = np.select(
            [reviews["compound"] >= 0.05, reviews["compound"] <= -0.05],
            ["Positive", "Negative"], default="Neutral",
        )
        self.assertTrue((reviews["sentiment_label"].to_numpy() == expected).all())
        # The source files were scraped separately. A few unlinked reviews are
        # permissible, but nearly all review rows must identify a bundled film.
        self.assertGreater(
            reviews["movie_id"].isin(self.movies["movie_id"]).mean(), 0.999,
        )


if __name__ == "__main__":
    unittest.main()
