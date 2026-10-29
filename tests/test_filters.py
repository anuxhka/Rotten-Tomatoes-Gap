import unittest

import numpy as np
import pandas as pd

from src.filters import genre_values, matching_titles


class FilterTests(unittest.TestCase):
    def test_parquet_genre_arrays_are_read(self):
        self.assertEqual(genre_values(np.array(["Drama", "Comedy"])), ["Drama", "Comedy"])
        self.assertEqual(genre_values(["Action", "Fantasy"]), ["Action", "Fantasy"])

    def test_literal_title_search_and_options(self):
        titles = pd.Series(["It", "it", "Little Women", "The IT Crowd", "A.B."])
        self.assertEqual(matching_titles(titles, "it").tolist(), [True, True, True, True, False])
        self.assertEqual(matching_titles(titles, "It", match_case=True).tolist(), [True, False, False, False, False])
        self.assertEqual(matching_titles(titles, "it", whole_word=True).tolist(), [True, True, False, True, False])
        self.assertEqual(matching_titles(titles, "A.B.", whole_word=True).tolist(), [False, False, False, False, True])


if __name__ == "__main__":
    unittest.main()
