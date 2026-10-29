"""Check the chart data shown for shared scores and signed gaps."""

import unittest

import pandas as pd

from src.charts import (
    create_comparison_bars,
    create_gap_distribution,
    create_gap_ranking,
    create_scatter_plot,
    create_sentiment_chart,
    create_sentiment_vs_rating,
)


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies = pd.DataFrame([
            {"movie_id": "a", "display_label": "A (2020)", "critic_rating": 80,
             "audience_rating": 60, "rating_gap": 20, "abs_gap": 20,
             "has_both_ratings": True},
            {"movie_id": "b", "display_label": "B (2020)", "critic_rating": 80,
             "audience_rating": 60, "rating_gap": 20, "abs_gap": 20,
             "has_both_ratings": True},
            {"movie_id": "c", "display_label": "C (2020)", "critic_rating": 25,
             "audience_rating": 75, "rating_gap": -50, "abs_gap": 50,
             "has_both_ratings": True},
        ])

    def test_scatter_aggregates_identical_score_pairs_and_highlights_selection(self):
        fig = create_scatter_plot(self.movies, selected_movies=["b"])
        self.assertEqual(len(fig.data[0].x), 2)
        self.assertEqual(sorted(fig.data[0].customdata[:, 0].tolist()), [1, 2])
        self.assertEqual(fig.data[1].text[0], "B (2020)")
        self.assertEqual(len(fig.layout.annotations), 4)
        self.assertEqual(len(create_scatter_plot(self.movies, show_quadrants=False).layout.annotations), 0)

    def test_ranking_direction_and_empty_state(self):
        critics = create_gap_ranking(self.movies, mode="critics_higher")
        audiences = create_gap_ranking(self.movies, mode="audiences_higher")
        self.assertEqual(list(critics.data[0].x), [20, 20])
        self.assertEqual(list(audiences.data[0].x), [-50])
        empty = create_gap_ranking(self.movies.iloc[:2], mode="audiences_higher")
        self.assertIn("No movies match", empty.layout.annotations[0].text)

    def test_other_charts_accept_filtered_and_empty_frames(self):
        self.assertEqual(len(create_comparison_bars(self.movies.iloc[:2]).data), 2)
        self.assertEqual(len(create_gap_distribution(self.movies).data), 1)
        self.assertEqual(len(create_gap_distribution(self.movies.iloc[:0]).data), 1)
        self.assertIn("No movies", create_scatter_plot(self.movies.iloc[:0]).layout.annotations[0].text)
        pie = create_sentiment_chart(60, 30, 10, 10)
        self.assertEqual(list(pie.data[0].values), [60, 30, 10])

    def test_sentiment_scatter_uses_webgl_and_keeps_each_film(self):
        movies = self.movies.assign(avg_sentiment=[0.2, -0.4, None],
                                    reviews_analyzed=[12, 3, None])
        fig = create_sentiment_vs_rating(movies)
        self.assertEqual(fig.data[0].type, "scattergl")
        self.assertEqual(list(fig.data[0].text), ["A (2020)", "B (2020)"])
        self.assertEqual(list(fig.data[0].customdata), [12, 3])


if __name__ == "__main__":
    unittest.main()
