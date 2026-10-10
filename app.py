"""
The Rotten Tomatoes Gap: Quantifying the Critic vs. Audience Disconnect

A Streamlit dashboard exploring differences between critic and audience
opinions using Stefano Leone's Rotten Tomatoes dataset from Kaggle.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path

from src.data_loader import (
    load_merged_data,
    load_reviews,
    get_all_genres,
    get_year_range,
)
from src.charts import (
    create_scatter_plot,
    create_gap_ranking,
    create_sentiment_chart,
    create_sentiment_vs_rating,
    create_comparison_bars,
    create_gap_distribution,
)


# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="The Rotten Tomatoes Gap",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    f"<style>{(Path(__file__).parent / 'assets' / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

HELP = {
    "search": "Search movie titles, ignoring capitalization. Punctuation is treated literally. This filters the overview, movie picker, and export.",
    "genres": "Show films tagged with any selected genre. Leave blank to include all genres.",
    "years": "Include films released within these years, including both endpoints. Unknown years appear only when the full range is selected.",
    "critic_count": "Minimum number of critics who rated a film in the movies dataset. Set to 0 for no minimum. This is different from the number of review texts analyzed below.",
    "audience_count": "Minimum number of audience ratings recorded for a film. Set to 0 for no minimum.",
    "critic_total": "Number of critics counted for this film's published critic rating. This can differ from the review texts available for sentiment analysis.",
    "audience_total": "Number of audience ratings counted for this film's published audience score.",
    "both": "Keep only films with both critic and audience percentage scores. Both are needed to calculate a rating gap.",
    "shortlist": "Save up to five films in this browser session. Remove films or clear the list in the Compare tab.",
    "snapshot": "A historical dataset scraped in October 2020. Scores and review counts are not live.",
    "films": "Number of films matching all current sidebar filters. Some may be missing one or both scores.",
    "rated": "Films in the current view that have both critic and audience percentage scores.",
    "average_gap": "Average absolute difference between critic and audience scores, in percentage points. Direction is ignored.",
    "largest_gap": "Biggest absolute critic–audience score difference in the current view. The film is shown underneath.",
    "map": "Each point is a film: audience score is horizontal, critic score vertical. On the diagonal, scores match; above it, critics score higher.",
    "quadrants": "Show 60% reference lines and labels for the four broad rating regions. These are visual guides, not certification labels.",
    "ranking": "Rank films by absolute gap, by critics scoring higher, or by audiences scoring higher. Only films with both scores appear.",
    "top_n": "Maximum number of films shown in the leaderboard. Fewer appear when the filters leave fewer matches.",
    "distribution": "Shows the signed gap: critic score minus audience score. Positive values mean critics score higher; negative values mean audiences score higher.",
    "movie_picker": "Choose from films matching the sidebar filters. Type a title to narrow the list.",
    "critic_score": "Percentage of critics whose reviews were classified as positive in the source dataset (Tomatometer).",
    "audience_score": "Percentage of audience ratings classified as positive in the source dataset.",
    "rating_gap": "Critic score minus audience score, in percentage points. Positive means critics scored higher.",
    "sentiment": "VADER analyzes critic review text only, not audience reviews. Its tone score may differ from the critic rating.",
    "compound": "Mean VADER compound score across available critic review texts, from −1 (negative) to +1 (positive).",
    "reviews_analyzed": "Number of individual critic review texts available and scored by VADER. It may differ from the published critic count.",
    "sentiment_share": "Share of analyzed critic review texts in this sentiment class. VADER uses ±0.05 as its positive and negative thresholds.",
    "review_filter": "Choose which VADER sentiment labels to show. Clear all to show no reviews.",
    "publication": "Limit the review table to selected publications. Leave blank to include every publication.",
    "review_type": "Choose Fresh and/or Rotten labels from the source review data. Clear all to show no reviews.",
    "comparison": "Compares the published critic and audience scores for films in your shortlist.",
    "quality": "Counts how many movie records have each field available in this historical snapshot.",
}

# ─── Session state initialization ──────────────────────────────
if "comparison_list" not in st.session_state:
    st.session_state.comparison_list = []
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None


# ─── Load data ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    movies = load_merged_data()
    reviews = load_reviews()
    return movies, reviews

movies_df, reviews_df = get_data()


# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">●</span>'
        '<div><strong>THE GAP</strong><small>ROTTEN TOMATOES / 2020</small></div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Movie Search ---
    st.subheader("Find a film", help=HELP["search"])
    search_query = st.text_input(
        "Search movie titles",
        placeholder="e.g., The Dark Knight",
        key="search_input",
        help=HELP["search"],
    )

    # --- Filters ---
    st.subheader("Refine results", help="These filters apply across the overview, movie picker, and full export.")

    # Genre filter
    all_genres = get_all_genres(movies_df)
    selected_genres = st.multiselect("Genres", options=all_genres, default=[], help=HELP["genres"])

    # Year range
    min_year, max_year = get_year_range(movies_df)
    year_range = st.slider(
        "Release Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help=HELP["years"],
    )

    # Minimum counts
    min_critic_count = st.number_input(
        "Min. Critic Count", min_value=0, value=0, step=5,
        help=HELP["critic_count"],
    )
    min_audience_count = st.number_input(
        "Min. Audience Count", min_value=0, value=0, step=100,
        help=HELP["audience_count"],
    )

    # Require both ratings toggle
    require_both = st.checkbox("Only movies with both ratings", value=False, help=HELP["both"])

    st.divider()

    # --- Dataset Info (compact) ---
    st.subheader("About the data", help=HELP["snapshot"])
    st.caption(f"**{len(movies_df):,}** movies · **{len(reviews_df):,}** critic reviews")
    st.caption("Source: Stefano Leone / Kaggle (CC0)")
    st.caption("Scraped: 2020-10-31 · Historical snapshot")


# ─── Apply filters ─────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the movies DataFrame."""
    filtered = df.copy()

    # Search
    if search_query:
        mask = filtered["title"].str.contains(search_query, case=False, na=False, regex=False)
        filtered = filtered[mask]

    # Genre
    if selected_genres:
        genre_mask = filtered["genres_list"].apply(
            lambda gl: any(g in gl for g in selected_genres)
            if isinstance(gl, list) else False
        )
        filtered = filtered[genre_mask]

    # Year
    year_mask = (
        (filtered["release_year"] >= year_range[0])
        & (filtered["release_year"] <= year_range[1])
    )
    if year_range == (min_year, max_year):
        year_mask |= filtered["release_year"].isna()
    filtered = filtered[year_mask]

    # Minimum counts
    if min_critic_count > 0:
        filtered = filtered[
            filtered["critic_count"].fillna(0) >= min_critic_count
        ]
    if min_audience_count > 0:
        filtered = filtered[
            filtered["audience_count"].fillna(0) >= min_audience_count
        ]

    # Require both ratings
    if require_both:
        filtered = filtered[filtered["has_both_ratings"]]

    return filtered


filtered_df = apply_filters(movies_df)


# ─── Helper functions ──────────────────────────────────────────
def format_rating(val) -> str:
    """Format a rating value, handling NaN."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.0f}%"


def format_gap(val) -> str:
    """Format a gap value with sign."""
    if pd.isna(val):
        return "N/A"
    return f"{val:+.0f} pp"


def format_count(val) -> str:
    """Format a count, handling NaN."""
    if pd.isna(val):
        return "N/A"
    return f"{val:,.0f}"


def safe_number(val) -> float:
    """Return zero for absent numeric values in sentiment summaries."""
    return 0.0 if pd.isna(val) else float(val)


def gap_color(val) -> str:
    """Return a color indicator for gap direction."""
    if pd.isna(val):
        return "⚪"
    if val > 10:
        return "🔴"  # Critics much higher
    elif val > 0:
        return "🟠"
    elif val < -10:
        return "🔵"  # Audiences much higher
    elif val < 0:
        return "🟢"
    return "⚪"


# ─── Main content area ────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-kicker"><span class="kicker-dot"></span> FILM OPINION, MAPPED</div>'
    '<h1>Where critics and audiences <em>part ways.</em></h1>'
    '<p>Explore the films that split opinion, then look closer at the reviews behind the scores.</p>'
    f'<div class="hero-foot">{len(movies_df):,} FILMS <span>•</span> CRITIC &amp; AUDIENCE RATINGS <span>•</span> OCT 2020 SNAPSHOT</div>'
    '</div>',
    unsafe_allow_html=True,
)
tab_overview, tab_detail, tab_compare, tab_info = st.tabs([
    "Overview", "Movie Detail", "Compare", "About the Data"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("The big picture", help="All four figures update with the sidebar filters. A rating gap needs both scores.")
    st.caption("Your sidebar search and filters apply to every chart and number in this view.")
    # Summary cards
    both_rated = filtered_df[filtered_df["has_both_ratings"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Films in view", f"{len(filtered_df):,}", help=HELP["films"])
    col2.metric("Both scores available", f"{len(both_rated):,}", help=HELP["rated"])

    if len(both_rated) > 0:
        mean_abs_gap = both_rated["abs_gap"].mean()
        col3.metric("Average disagreement", f"{mean_abs_gap:.1f} pp", help=HELP["average_gap"])

        max_gap_row = both_rated.loc[both_rated["abs_gap"].idxmax()]
        col4.metric(
            "Largest disagreement",
            f"{max_gap_row['abs_gap']:.0f} pp",
            delta=max_gap_row["display_label"],
            delta_color="off",
            help=HELP["largest_gap"],
        )
    else:
        col3.metric("Average disagreement", "N/A", help=HELP["average_gap"])
        col4.metric("Largest disagreement", "N/A", help=HELP["largest_gap"])

    if both_rated.empty:
        st.info("No films with both ratings match these filters. Broaden your search to see the charts.")
    else:
        st.subheader("The opinion map", help=HELP["map"])
        st.caption("Each dot is a film. Distance from the diagonal shows the size of the rating gap.")

        col_scatter, col_controls = st.columns([5, 1])
        with col_controls:
            show_quadrants = st.checkbox("Show quadrant labels", value=True, help=HELP["quadrants"])
            st.caption("Above the diagonal: critics higher. Below: audiences higher.")

        with col_scatter:
            fig_scatter = create_scatter_plot(
                filtered_df,
                show_quadrants=show_quadrants,
                selected_movies=st.session_state.comparison_list,
            )
            st.plotly_chart(fig_scatter, width="stretch", key="main_scatter")

        st.divider()
        st.subheader("The disagreement leaderboard", help=HELP["ranking"])
        control_mode, control_count = st.columns([2, 1])
        with control_mode:
            rank_mode = st.selectbox(
                "Ranking mode",
                options=["largest", "critics_higher", "audiences_higher"],
                format_func=lambda x: {
                    "largest": "Largest Disagreement",
                    "critics_higher": "Critics Rate Higher",
                    "audiences_higher": "Audiences Rate Higher",
                }[x],
                help=HELP["ranking"],
            )
        with control_count:
            top_n = st.slider("Number of films", 10, 30, 15, key="rank_topn", help=HELP["top_n"])
        fig_rank = create_gap_ranking(filtered_df, mode=rank_mode, top_n=top_n)
        st.plotly_chart(fig_rank, width="stretch")

        st.subheader("How gaps are distributed", help=HELP["distribution"])
        fig_dist = create_gap_distribution(filtered_df)
        st.plotly_chart(fig_dist, width="stretch")

        critics_higher_pct = (both_rated["rating_gap"] > 0).mean() * 100
        audiences_higher_pct = (both_rated["rating_gap"] < 0).mean() * 100
        agree_pct = (both_rated["rating_gap"] == 0).mean() * 100

        st.caption(
            f"Mean gap: **{both_rated['rating_gap'].mean():+.1f} pp** · "
            f"Critics higher: **{critics_higher_pct:.1f}%** of films · "
            f"Audiences higher: **{audiences_higher_pct:.1f}%** · "
            f"Equal: **{agree_pct:.1f}%**"
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2: MOVIE DETAIL
# ═══════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("Go beyond the scores", help="Inspect published ratings and available critic review text for one film.")
    st.caption("Choose a film from your current filters to inspect its ratings and critic reviews.")
    search_results = filtered_df

    if len(search_results) == 0:
        st.info("No films match your current filters. Broaden them in the sidebar.")
    else:
        sorted_movies = search_results.sort_values(["display_label", "movie_id"])
        movie_options = sorted_movies["movie_id"].tolist()
        movie_labels = dict(zip(sorted_movies["movie_id"], sorted_movies["display_label"]))
        selected_id = st.selectbox(
            "Choose a movie",
            options=movie_options,
            index=None,
            placeholder="Search or choose a film…",
            format_func=lambda movie_id: movie_labels[movie_id],
            key="movie_selector",
            help=HELP["movie_picker"],
        )

        if selected_id:
            movie_row = search_results[search_results["movie_id"] == selected_id].iloc[0]
            movie_id = selected_id

            # Add to comparison button
            if movie_id not in st.session_state.comparison_list:
                if st.button("Add to comparison", key="add_compare", disabled=len(st.session_state.comparison_list) >= 5,
                             help=HELP["shortlist"]):
                    st.session_state.comparison_list.append(movie_id)
                    st.rerun()
                if len(st.session_state.comparison_list) >= 5:
                    st.caption("Comparison is full. Remove a film in the Compare tab to add this one.")
            else:
                st.success("✓ In comparison list")

            st.divider()

            # --- Movie metadata ---
            col_meta, col_ratings = st.columns([2, 1])

            with col_meta:
                st.subheader(f"🎬 {movie_row.get('title', 'Unknown')}")

                meta_items = []
                if pd.notna(movie_row.get("release_year")):
                    meta_items.append(f"**Year:** {int(movie_row['release_year'])}")
                if pd.notna(movie_row.get("genres")):
                    meta_items.append(f"**Genres:** {movie_row['genres']}")
                if pd.notna(movie_row.get("content_rating")):
                    meta_items.append(f"**Rated:** {movie_row['content_rating']}")
                if pd.notna(movie_row.get("runtime_minutes")):
                    meta_items.append(f"**Runtime:** {int(movie_row['runtime_minutes'])} min")
                if pd.notna(movie_row.get("directors")):
                    meta_items.append(f"**Director:** {movie_row['directors']}")

                st.markdown(" · ".join(meta_items) if meta_items else "No metadata available")

                if pd.notna(movie_row.get("synopsis")):
                    with st.expander("Synopsis"):
                        st.write(movie_row["synopsis"])

            with col_ratings:
                st.subheader("Ratings", help="The source dataset's published percentage scores and rating counts for this film.")

                r_col1, r_col2 = st.columns(2)
                r_col1.metric("🍅 Critic", format_rating(movie_row.get("critic_rating")), help=HELP["critic_score"])
                r_col2.metric("🍿 Audience", format_rating(movie_row.get("audience_rating")), help=HELP["audience_score"])

                gap_val = movie_row.get("rating_gap")
                st.metric(
                    f"{gap_color(gap_val)} Rating Gap",
                    format_gap(gap_val),
                    help=HELP["rating_gap"],
                )

                c_col1, c_col2 = st.columns(2)
                c_col1.metric("Critic count", format_count(movie_row.get("critic_count")), help=HELP["critic_total"])
                c_col2.metric("Audience count", format_count(movie_row.get("audience_count")), help=HELP["audience_total"])

            st.divider()

            # --- Critic Review Sentiment ---
            st.subheader("📝 Critic Review Text Sentiment (VADER)", help=HELP["sentiment"])
            st.caption(
                "Sentiment analysis of **critic review text only**. "
                "This is NOT audience sentiment. "
                "VADER scores may misinterpret sarcasm or mixed opinions."
            )

            movie_reviews = reviews_df[reviews_df["movie_id"] == movie_id]

            if len(movie_reviews) == 0:
                st.info("No critic reviews with text available for this movie. "
                       "Rating comparison above uses the dataset's aggregate ratings.")
            else:
                col_sent_chart, col_sent_stats = st.columns([2, 1])

                with col_sent_chart:
                    pct_pos = safe_number(movie_row.get("pct_positive"))
                    pct_neu = safe_number(movie_row.get("pct_neutral"))
                    pct_neg = safe_number(movie_row.get("pct_negative"))
                    n_analyzed = int(safe_number(movie_row.get("reviews_analyzed")))

                    fig_sent = create_sentiment_chart(
                        pct_pos, pct_neu, pct_neg, n_analyzed,
                        title=f"Critic Review Sentiment — {movie_row.get('title', '')}"
                    )
                    st.plotly_chart(fig_sent, width="stretch")

                with col_sent_stats:
                    avg_sent = movie_row.get("avg_sentiment")
                    st.metric(
                        "Avg. Compound Score",
                        f"{avg_sent:.3f}" if pd.notna(avg_sent) else "N/A",
                        help=HELP["compound"],
                    )
                    st.metric("Reviews Analyzed", f"{n_analyzed:,}", help=HELP["reviews_analyzed"])
                    st.metric("Positive", f"{pct_pos:.1f}%", help=HELP["sentiment_share"])
                    st.metric("Neutral", f"{pct_neu:.1f}%", help=HELP["sentiment_share"])
                    st.metric("Negative", f"{pct_neg:.1f}%", help=HELP["sentiment_share"])

                st.divider()

                # --- Review Explorer ---
                st.subheader("📖 Review Explorer", help="Browse individual critic review texts and their VADER sentiment labels.")

                # Review filters
                rev_col1, rev_col2, rev_col3 = st.columns(3)
                with rev_col1:
                    sent_filter = st.multiselect(
                        "Sentiment",
                        options=["Positive", "Neutral", "Negative"],
                        default=["Positive", "Neutral", "Negative"],
                        key="sent_filter",
                        help=HELP["review_filter"],
                    )
                with rev_col2:
                    pubs = movie_reviews["publication"].dropna().unique().tolist()
                    pub_filter = st.multiselect(
                        "Publication", options=sorted(pubs), default=[], key="pub_filter",
                        help=HELP["publication"],
                    )
                with rev_col3:
                    rev_type_filter = st.multiselect(
                        "Review Type",
                        options=["Fresh", "Rotten"],
                        default=["Fresh", "Rotten"],
                        key="rev_type_filter",
                        help=HELP["review_type"],
                    )

                # Apply review filters
                display_reviews = movie_reviews.copy()
                display_reviews = display_reviews[
                    display_reviews["sentiment_label"].isin(sent_filter)
                ]
                if pub_filter:
                    display_reviews = display_reviews[
                        display_reviews["publication"].isin(pub_filter)
                    ]
                display_reviews = display_reviews[
                    display_reviews["review_type"].isin(rev_type_filter)
                ]

                st.caption(f"Showing {len(display_reviews)} of {len(movie_reviews)} reviews")

                # Display reviews
                if len(display_reviews) > 0:
                    review_display = display_reviews[[
                        "critic_name", "publication", "review_date",
                        "review_type", "compound", "sentiment_label", "review_text"
                    ]].copy()
                    review_display.columns = [
                        "Critic", "Publication", "Date",
                        "Type", "Score", "Sentiment", "Review"
                    ]
                    review_display["Date"] = review_display["Date"].dt.strftime("%Y-%m-%d").fillna("N/A")
                    review_display["Score"] = review_display["Score"].round(3)
                    review_display = review_display.sort_values("Score", ascending=False)

                    st.dataframe(
                        review_display,
                        width="stretch",
                        hide_index=True,
                        height=400,
                        column_config={
                            "Critic": st.column_config.TextColumn(help="Name credited for the review in the source data."),
                            "Publication": st.column_config.TextColumn(help="Publication where the critic review appeared."),
                            "Date": st.column_config.TextColumn(help="Review date, when available in the source data."),
                            "Type": st.column_config.TextColumn(help="Fresh or Rotten label in the source review data."),
                            "Score": st.column_config.NumberColumn(format="%.3f", help="VADER compound tone score for this review, from −1 to +1."),
                            "Sentiment": st.column_config.TextColumn(help="VADER label based on the compound score: positive above 0.05, negative below −0.05, neutral between."),
                            "Review": st.column_config.TextColumn(width="large", help="The critic review text analyzed by VADER."),
                        },
                    )
                else:
                    st.info("No reviews match the current filters.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: COMPARE
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare your shortlist", help=HELP["comparison"])
    st.caption("Put up to five films side by side to see where ratings and critic review tone differ.")

    if not st.session_state.comparison_list:
        st.info(
            "No movies in your comparison list yet.\n\n"
            "Go to **Movie Detail**, choose a film, and click "
            "**Add to comparison** to build your shortlist."
        )
    else:
        comp_df = movies_df[movies_df["movie_id"].isin(st.session_state.comparison_list)]

        if len(comp_df) == 0:
            st.warning("Selected movies not found in dataset.")
        else:
            st.subheader("Your films", help=HELP["shortlist"])
            with st.expander(f"Edit shortlist · {len(comp_df)}/5 films"):
                for _, row in comp_df.iterrows():
                    if st.button(f"Remove {row['display_label']}", key=f"remove_{row['movie_id']}",
                                 help="Remove this film from the comparison."):
                        st.session_state.comparison_list.remove(row["movie_id"])
                        st.rerun()
                if st.button("Clear shortlist", help="Remove all films from the comparison."):
                    st.session_state.comparison_list = []
                    st.rerun()

            # Comparison table
            st.subheader("Rating comparison", help="Side-by-side critic and audience scores, their signed gap, and rating counts.")
            comp_table = comp_df[[
                "display_label", "critic_rating", "audience_rating",
                "rating_gap", "critic_count", "audience_count"
            ]].copy()
            comp_table.columns = [
                "Movie", "Critic %", "Audience %", "Gap (pp)",
                "Critic Count", "Audience Count"
            ]
            comp_table = comp_table.sort_values("Gap (pp)", ascending=False, na_position="last")

            st.dataframe(
                comp_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Movie": st.column_config.TextColumn(help="Film title and release year."),
                    "Critic %": st.column_config.NumberColumn(format="%.0f%%", help=HELP["critic_score"]),
                    "Audience %": st.column_config.NumberColumn(format="%.0f%%", help=HELP["audience_score"]),
                    "Gap (pp)": st.column_config.NumberColumn(format="%+.0f", help=HELP["rating_gap"]),
                    "Critic Count": st.column_config.NumberColumn(format="%,.0f", help=HELP["critic_total"]),
                    "Audience Count": st.column_config.NumberColumn(format="%,.0f", help=HELP["audience_total"]),
                },
            )

            # Comparison bar chart
            fig_comp = create_comparison_bars(comp_df)
            st.plotly_chart(fig_comp, width="stretch")

            # Sentiment comparison
            st.subheader("Critic review sentiment comparison", help=HELP["sentiment"])
            st.caption("VADER sentiment of critic review text only.")

            sent_comp = comp_df[[
                "display_label", "avg_sentiment", "reviews_analyzed",
                "pct_positive", "pct_neutral", "pct_negative"
            ]].copy()
            sent_comp.columns = [
                "Movie", "Avg. Sentiment", "Reviews Analyzed",
                "% Positive", "% Neutral", "% Negative"
            ]

            st.dataframe(
                sent_comp,
                width="stretch",
                hide_index=True,
                column_config={
                    "Movie": st.column_config.TextColumn(help="Film title and release year."),
                    "Avg. Sentiment": st.column_config.NumberColumn(format="%.3f", help=HELP["compound"]),
                    "Reviews Analyzed": st.column_config.NumberColumn(help=HELP["reviews_analyzed"]),
                    "% Positive": st.column_config.NumberColumn(format="%.1f%%", help=HELP["sentiment_share"]),
                    "% Neutral": st.column_config.NumberColumn(format="%.1f%%", help=HELP["sentiment_share"]),
                    "% Negative": st.column_config.NumberColumn(format="%.1f%%", help=HELP["sentiment_share"]),
                },
            )

            st.divider()

            # CSV Export
            st.subheader("Export your shortlist", help="Download one CSV row per film in this comparison, including ratings and critic text sentiment.")
            export_df = comp_df[[
                "title", "release_year", "genres", "critic_rating", "audience_rating",
                "rating_gap", "abs_gap", "critic_count", "audience_count",
                "avg_sentiment", "reviews_analyzed", "pct_positive", "pct_neutral", "pct_negative"
            ]].copy()
            export_df.columns = [
                "Title", "Year", "Genres", "Critic Rating (%)", "Audience Rating (%)",
                "Rating Gap (pp)", "Absolute Gap (pp)", "Critic Count", "Audience Count",
                "Avg Critic Text Sentiment", "Reviews Analyzed",
                "% Positive Reviews", "% Neutral Reviews", "% Negative Reviews"
            ]

            csv_buffer = BytesIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Comparison as CSV",
                data=csv_buffer.getvalue(),
                file_name="rotten_tomatoes_comparison.csv",
                mime="text/csv",
                width="stretch",
                help="Save the current comparison as a CSV file.",
            )


# ═══════════════════════════════════════════════════════════════
# TAB 4: DATASET INFO
# ═══════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("What you're looking at", help=HELP["snapshot"])

    st.markdown("""
    ### Source

    This dashboard uses **Stefano Leone's Rotten Tomatoes Movies and Critic Reviews Dataset**,
    available on [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
    under the **CC0: Public Domain** license.

    ### Data Snapshot

    The data was scraped from [rottentomatoes.com](https://www.rottentomatoes.com) as of **October 31, 2020**.
    All ratings, reviews, and metadata reflect that point in time. Current Rotten Tomatoes pages may differ.

    ### Coverage
    """)

    total_movies = len(movies_df)
    both_rated_total = movies_df["has_both_ratings"].sum()
    total_reviews = len(reviews_df)
    movies_with_reviews = reviews_df["movie_id"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{total_movies:,}", help="All film records in the source snapshot, before sidebar filters.")
    col2.metric("With Both Ratings", f"{both_rated_total:,}", help="Film records with both published critic and audience percentage scores.")
    col3.metric("Total Critic Reviews", f"{total_reviews:,}", help="Individual critic review records with text in the reviews file.")
    col4.metric("Movies with Reviews", f"{movies_with_reviews:,}", help="Films linked to at least one individual critic review record.")

    st.markdown("""
    ### Important Notes

    - **No audience text reviews.** This dataset contains individual critic review text but
      only aggregate audience rating percentages. The dashboard never fabricates audience text sentiment.
    - **Critic text sentiment ≠ Critic rating.** VADER scores measure the sentiment of review text.
      The Tomatometer (critic rating %) measures the proportion of positive reviews. These are related
      but not identical measures.
    - **Historical data.** These are not live scores. A movie's current Rotten Tomatoes rating
      may have changed since October 2020.
    - **Scraping consistency.** Because movies and reviews were scraped in stages, aggregate counts
      in the movies file may not exactly match the number of individual reviews in the reviews file.
    - **Sarcasm and nuance.** VADER may misinterpret sarcasm, irony, or mixed critical opinions.
      Sentiment results are indicative, not definitive.

    ### Rating Gap Definition

    The **Rating Gap** is defined as:

    > **Gap = Critic Rating (%) − Audience Rating (%)**

    | Gap Value | Interpretation |
    |:--|:--|
    | Positive | Critics rate higher than audiences |
    | Negative | Audiences rate higher than critics |
    | Zero | Both published percentages match |

    Range: −100 to +100 percentage points.
    """)

    st.divider()

    # Sentiment vs Rating scatter
    st.subheader("Critic text sentiment vs. critic rating", help="Compares average VADER tone of available critic review texts with the published critic percentage score.")
    st.caption(
        "Comparing VADER compound scores of critic review text against the aggregate "
        "Tomatometer rating. A film can have positive review tone and a lower "
        "Tomatometer (or the reverse), because these measures answer different questions."
    )
    fig_svr = create_sentiment_vs_rating(movies_df)
    st.plotly_chart(fig_svr, width="stretch")

    st.divider()

    # Data quality overview
    st.subheader("📋 Data Quality Summary", help=HELP["quality"])

    quality_data = {
        "Field": ["Critic Rating", "Audience Rating", "Release Year", "Genres", "Runtime"],
        "Available": [
            f"{movies_df['critic_rating'].notna().sum():,}",
            f"{movies_df['audience_rating'].notna().sum():,}",
            f"{movies_df['release_year'].notna().sum():,}",
            f"{(movies_df['genres'].notna() & (movies_df['genres'] != '')).sum():,}",
            f"{movies_df.get('runtime_minutes', pd.Series(dtype='float64')).notna().sum():,}",
        ],
        "Missing": [
            f"{movies_df['critic_rating'].isna().sum():,}",
            f"{movies_df['audience_rating'].isna().sum():,}",
            f"{movies_df['release_year'].isna().sum():,}",
            f"{(movies_df['genres'].isna() | (movies_df['genres'] == '')).sum():,}",
            f"{movies_df.get('runtime_minutes', pd.Series(dtype='float64')).isna().sum():,}",
        ],
        "% Available": [
            f"{movies_df['critic_rating'].notna().mean()*100:.1f}%",
            f"{movies_df['audience_rating'].notna().mean()*100:.1f}%",
            f"{movies_df['release_year'].notna().mean()*100:.1f}%",
            f"{(movies_df['genres'].notna() & (movies_df['genres'] != '')).mean()*100:.1f}%",
            f"{movies_df.get('runtime_minutes', pd.Series(dtype='float64')).notna().mean()*100:.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(quality_data), hide_index=True, width="stretch")

    # Full filtered export
    st.divider()
    st.subheader("Export filtered films", help="Download one CSV row per film matching your current sidebar filters.")

    export_all = filtered_df[[
        "title", "release_year", "genres", "critic_rating", "audience_rating",
        "rating_gap", "abs_gap", "critic_count", "audience_count",
    ]].copy()
    export_all.columns = [
        "Title", "Year", "Genres", "Critic Rating (%)", "Audience Rating (%)",
        "Rating Gap (pp)", "Absolute Gap (pp)", "Critic Count", "Audience Count",
    ]

    csv_all = BytesIO()
    export_all.to_csv(csv_all, index=False)
    st.download_button(
        label=f"⬇️ Download {len(export_all):,} Filtered Movies as CSV",
        data=csv_all.getvalue(),
        file_name="rotten_tomatoes_filtered.csv",
        mime="text/csv",
        width="stretch",
        help="Save the currently filtered movie records as a CSV file.",
    )
