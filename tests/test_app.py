"""Run the dashboard headlessly with Streamlit's AppTest and drive its controls."""

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
TOTAL_FILMS = "17,712"


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        movies = pd.read_parquet(
            ROOT / "data" / "processed" / "movies_clean.parquet",
            columns=["movie_id", "critic_count"],
        )
        # The most-reviewed films always have critic reviews to explore.
        cls.films = movies.nlargest(6, "critic_count")["movie_id"].tolist()

    def setUp(self):
        self.at = AppTest.from_file(str(APP), default_timeout=240)
        self.run_app()

    def run_app(self, tab=None, change=None):
        """Apply a widget change and rerun. AppTest cannot click tabs, so the tab is pinned in session state."""
        if change:
            change()
        if tab:
            self.at.session_state["main_tabs"] = tab
        self.at.run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])
        self.assertFalse(self.at.error, [e.value for e in self.at.error])

    def films_in_view(self):
        return self.at.metric[0].value

    def test_overview_metrics(self):
        metrics = {m.label: m.value for m in self.at.metric}
        self.assertEqual(metrics["Films in view"], TOTAL_FILMS)
        self.assertEqual(metrics["Both scores available"], "17,407")
        self.assertEqual(metrics["Average disagreement"], "16.9 pp")

    def test_title_search_and_options(self):
        self.run_app(change=lambda: self.at.text_input(key="search_input").set_value("Star Wars"))
        films = int(self.films_in_view().replace(",", ""))
        self.assertTrue(0 < films < 100, films)
        self.run_app(change=lambda: self.at.checkbox(key="whole_word").check())
        self.run_app(change=lambda: self.at.checkbox(key="match_case").check())
        self.assertGreater(int(self.films_in_view()), 0)
        self.run_app(change=lambda: self.at.text_input(key="search_input").set_value("zzqqxx-no-such-film"))
        self.assertEqual(self.films_in_view(), "0")

    def test_filters_and_reset(self):
        self.run_app(change=lambda: self.at.multiselect(key="genre_filter").set_value(["Documentary"]))
        self.run_app(change=lambda: self.at.slider(key="year_filter").set_value((2000, 2010)))
        self.run_app(change=lambda: self.at.number_input(key="critic_filter").set_value(20))
        self.run_app(change=lambda: self.at.checkbox(key="both_filter").check())
        self.assertNotEqual(self.films_in_view(), TOTAL_FILMS)
        reset = next(b for b in self.at.button if b.label == "Reset filters")
        self.run_app(change=reset.click)
        self.assertEqual(self.films_in_view(), TOTAL_FILMS)

    def test_ranking_modes(self):
        def ranking_mode():
            return next(s for s in self.at.selectbox if s.label == "Ranking mode")

        for option in ["largest", "critics_higher", "audiences_higher"]:
            self.run_app(change=lambda o=option: ranking_mode().set_value(o))
            self.assertEqual(ranking_mode().value, option)
        self.run_app(change=lambda: self.at.slider(key="rank_topn").set_value(30))

    def test_movie_detail_and_review_filters(self):
        self.run_app(tab="Movie Detail")
        self.assertEqual(len(self.at.selectbox(key="movie_selector").options), 17712)
        self.run_app("Movie Detail", lambda: self.at.selectbox(key="movie_selector").set_value(self.films[0]))
        labels = [m.label for m in self.at.metric]
        self.assertTrue(any("Rating Gap" in label for label in labels), labels)
        self.assertIn("Reviews Analyzed", labels)
        self.run_app("Movie Detail", lambda: self.at.multiselect(key="sent_filter").set_value(["Negative"]))
        self.run_app("Movie Detail", lambda: self.at.multiselect(key="rev_type_filter").set_value(["Rotten"]))
        self.run_app("Movie Detail", lambda: self.at.multiselect(key="sent_filter").set_value([]))

    def test_comparison_shortlist(self):
        self.run_app(tab="Movie Detail")
        for film in self.films:
            self.run_app("Movie Detail", lambda f=film: self.at.selectbox(key="movie_selector").set_value(f))
            keys = [b.key for b in self.at.button]
            if "add_compare" in keys and not self.at.button(key="add_compare").disabled:
                self.run_app("Movie Detail", lambda: self.at.button(key="add_compare").click())
        self.assertEqual(len(self.at.session_state.comparison_list), 5)
        self.assertTrue(self.at.button(key="add_compare").disabled, "a sixth film must be blocked")

        self.run_app(tab="Compare")
        removes = [b for b in self.at.button if (b.key or "").startswith("remove_")]
        self.assertEqual(len(removes), 5)
        self.run_app("Compare", removes[0].click)
        self.assertEqual(len(self.at.session_state.comparison_list), 4)
        clear = next(b for b in self.at.button if b.label == "Clear shortlist")
        self.run_app("Compare", clear.click)
        self.assertEqual(self.at.session_state.comparison_list, [])

    def test_about_tab_shows_data_disclosure(self):
        self.run_app(tab="About the Data")
        text = " ".join(m.value for m in self.at.markdown)
        self.assertIn("Kaggle", text)


if __name__ == "__main__":
    unittest.main()
