"""Small end-to-end fixtures for the preprocessing and sentiment pipeline."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_movies, clean_reviews, validate_join
from src.sentiment import aggregate_movie_sentiment, score_reviews


class PipelineTests(unittest.TestCase):
    def test_empty_movies_csv_fails_clearly(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "empty_movies.csv"
            pd.DataFrame(columns=[
                "rotten_tomatoes_link", "movie_title", "genres",
                "tomatometer_rating", "audience_rating", "tomatometer_count",
                "audience_count",
            ]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Movies CSV contains no rows"):
                clean_movies(path)

    def test_invalid_movie_identity_fails_before_export(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "movies.csv"
            base = {
                "movie_title": "A Film", "genres": "Drama",
                "tomatometer_rating": 80, "audience_rating": 60,
                "tomatometer_count": 10, "audience_count": 20,
            }
            for rows, error in [
                ([dict(base, rotten_tomatoes_link=None)], "missing movie IDs"),
                ([dict(base, rotten_tomatoes_link="m/a", movie_title="  ")], "missing titles"),
                ([dict(base, rotten_tomatoes_link="m/a"),
                  dict(base, rotten_tomatoes_link=" m/a ")], "duplicate movie IDs"),
            ]:
                pd.DataFrame(rows).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, error):
                    clean_movies(path)

    def test_reviews_without_movie_ids_are_excluded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "reviews.csv"
            pd.DataFrame([
                {"rotten_tomatoes_link": "m/a", "critic_name": "A",
                 "publisher_name": "Paper", "review_date": "2020-01-01",
                 "review_type": "Fresh", "review_content": "Good film."},
                {"rotten_tomatoes_link": None, "critic_name": "B",
                 "publisher_name": "Paper", "review_date": "2020-01-01",
                 "review_type": "Fresh", "review_content": "Another good film."},
            ]).to_csv(path, index=False)
            reviews = clean_reviews(path)
        self.assertEqual(reviews["movie_id"].tolist(), ["m/a"])

    def test_mixed_source_date_formats_are_kept(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            movies_path = Path(directory) / "movies.csv"
            reviews_path = Path(directory) / "reviews.csv"
            base = {
                "genres": "Drama", "tomatometer_rating": 80,
                "audience_rating": 60, "tomatometer_count": 10,
                "audience_count": 20,
            }
            pd.DataFrame([
                dict(base, rotten_tomatoes_link="m/a", movie_title="A",
                     original_release_date="2020-01-01"),
                dict(base, rotten_tomatoes_link="m/b", movie_title="B",
                     original_release_date="02/13/2020"),
            ]).to_csv(movies_path, index=False)
            pd.DataFrame([
                {"rotten_tomatoes_link": "m/a", "critic_name": "A",
                 "publisher_name": "Paper", "review_date": "2020-01-01",
                 "review_type": "Fresh", "review_content": "Good."},
                {"rotten_tomatoes_link": "m/b", "critic_name": "B",
                 "publisher_name": "Paper", "review_date": "02/13/2020",
                 "review_type": "Fresh", "review_content": "Also good."},
            ]).to_csv(reviews_path, index=False)
            movies = clean_movies(movies_path)
            reviews = clean_reviews(reviews_path)
        self.assertEqual(movies["release_year"].fillna(-1).tolist(), [2020, 2020])
        self.assertTrue(reviews["review_date"].notna().all())

    def test_empty_reviews_can_be_scored_and_aggregated(self):
        empty = pd.DataFrame({
            "movie_id": pd.Series(dtype="str"),
            "review_text": pd.Series(dtype="str"),
        })
        scored = score_reviews(empty)
        summary = aggregate_movie_sentiment(scored)
        self.assertTrue(scored.empty)
        self.assertEqual(scored["compound"].dtype, "float64")
        self.assertTrue(summary.empty)
        self.assertIn("pct_positive", summary.columns)

    def test_missing_required_source_columns_fail_with_a_clear_error(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "incomplete.csv"
            pd.DataFrame([{"movie_title": "Missing ID"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Movies CSV is missing required columns:.*movie_id"):
                clean_movies(path)
            with self.assertRaisesRegex(ValueError, "Reviews CSV is missing required columns:.*review_text"):
                clean_reviews(path)

    def test_cleaning_scoring_and_join(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            movies_path = Path(directory) / "movies.csv"
            reviews_path = Path(directory) / "reviews.csv"
            pd.DataFrame([
                {
                    "rotten_tomatoes_link": "m1", "movie_title": "A Film",
                    "original_release_date": "2020-01-02", "genres": "Drama, Comedy",
                    "tomatometer_rating": "80", "audience_rating": "60",
                    "tomatometer_count": "10", "audience_count": "20",
                    "movie_info": "A real description.", "runtime": "95 minutes",
                },
                {
                    "rotten_tomatoes_link": "m2", "movie_title": "Another Film",
                    "original_release_date": "bad date", "genres": None,
                    "tomatometer_rating": "101", "audience_rating": "-1",
                    "tomatometer_count": -3, "audience_count": -2,
                    "movie_info": None, "runtime": None,
                },
            ]).to_csv(movies_path, index=False)
            pd.DataFrame([
                {"rotten_tomatoes_link": "m1", "critic_name": "A",
                 "review_content": "Wonderful and delightful!", "review_type": "Fresh",
                 "publisher_name": "Paper", "review_date": "2020-01-03"},
                {"rotten_tomatoes_link": "m1", "critic_name": "A",
                 "review_content": "Wonderful and delightful!", "review_type": "Fresh",
                 "publisher_name": "Paper", "review_date": "2020-01-03"},
                {"rotten_tomatoes_link": "m2", "critic_name": "B",
                 "review_content": "Terrible and boring.", "review_type": "Rotten",
                 "publisher_name": "Journal", "review_date": "bad date"},
                {"rotten_tomatoes_link": "m2", "critic_name": "C",
                 "review_content": "   ", "review_type": "Fresh",
                 "publisher_name": "Journal", "review_date": None},
            ]).to_csv(reviews_path, index=False)

            movies = clean_movies(movies_path)
            reviews = clean_reviews(reviews_path)
            scored = score_reviews(reviews)
            summary = aggregate_movie_sentiment(scored)

        self.assertEqual(len(movies), 2)
        self.assertEqual(movies.loc[0, "rating_gap"], 20)
        self.assertEqual(movies.loc[0, "genres_list"], ["Drama", "Comedy"])
        self.assertEqual(movies.loc[0, "runtime_minutes"], 95)
        self.assertEqual(movies.loc[0, "synopsis"], "A real description.")
        self.assertFalse(bool(movies.loc[1, "has_both_ratings"]))
        self.assertTrue(pd.isna(movies.loc[1, "critic_rating"]))
        self.assertTrue(pd.isna(movies.loc[1, "audience_rating"]))
        self.assertTrue(pd.isna(movies.loc[1, "critic_count"]))
        self.assertTrue(pd.isna(movies.loc[1, "audience_count"]))
        self.assertEqual(len(reviews), 2)
        self.assertEqual(validate_join(movies, reviews)["movies_with_reviews"], 2)
        self.assertEqual(set(scored["sentiment_label"]), {"Positive", "Negative"})
        self.assertEqual(set(summary["reviews_analyzed"]), {1})
        self.assertTrue((summary[["pct_positive", "pct_neutral", "pct_negative"]].sum(axis=1) == 100).all())


if __name__ == "__main__":
    unittest.main()
