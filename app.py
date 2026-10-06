"""
The Rotten Tomatoes Gap: Quantifying the Critic vs. Audience Disconnect

A Streamlit dashboard exploring differences between critic and audience
opinions using Stefano Leone's Rotten Tomatoes dataset from Kaggle.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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
    initial_sidebar_state="expanded",
)

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
    st.title("🍅 The Rotten Tomatoes Gap")
    st.caption("Quantifying the Critic vs. Audience Disconnect")

    st.divider()

    # --- Movie Search ---
    st.subheader("🔍 Movie Search")
    search_query = st.text_input(
        "Search movie titles",
        placeholder="e.g., The Dark Knight",
        key="search_input",
    )

    # --- Filters ---
    st.subheader("🎛️ Filters")

    # Genre filter
    all_genres = get_all_genres(movies_df)
    selected_genres = st.multiselect("Genres", options=all_genres, default=[])

    # Year range
    min_year, max_year = get_year_range(movies_df)
    year_range = st.slider(
        "Release Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    # Minimum counts
    min_critic_count = st.number_input(
        "Min. Critic Count", min_value=0, value=0, step=5
    )
    min_audience_count = st.number_input(
        "Min. Audience Count", min_value=0, value=0, step=100
    )

    # Require both ratings toggle
    require_both = st.checkbox("Only movies with both ratings", value=False)

    st.divider()

    # --- Comparison List ---
    st.subheader("⚖️ Comparison List")
    if st.session_state.comparison_list:
        comp_movies = movies_df[movies_df["movie_id"].isin(st.session_state.comparison_list)]
        for _, row in comp_movies.iterrows():
            col1, col2 = st.columns([4, 1])
            col1.write(f"📽️ {row['display_label']}")
            if col2.button("✕", key=f"remove_{row['movie_id']}"):
                st.session_state.comparison_list.remove(row["movie_id"])
                st.rerun()
        if st.button("Clear All", use_container_width=True):
            st.session_state.comparison_list = []
            st.rerun()
    else:
        st.caption("No movies added yet. Use the Movie Detail tab to add.")

    st.divider()

    # --- Dataset Info (compact) ---
    st.subheader("ℹ️ Dataset")
    st.caption(f"**{len(movies_df):,}** movies · **{len(reviews_df):,}** critic reviews")
    st.caption("Source: Stefano Leone / Kaggle (CC0)")
    st.caption("Scraped: 2020-10-31 · Historical snapshot")


# ─── Apply filters ─────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the movies DataFrame."""
    filtered = df.copy()

    # Search
    if search_query:
        mask = filtered["title"].str.contains(search_query, case=False, na=False)
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
    ) | filtered["release_year"].isna()
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
tab_overview, tab_detail, tab_compare, tab_info = st.tabs([
    "📊 Overview", "🎬 Movie Detail", "⚖️ Compare", "ℹ️ Dataset Info"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    # Summary cards
    both_rated = filtered_df[filtered_df["has_both_ratings"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies Matching", f"{len(filtered_df):,}")
    col2.metric("With Both Ratings", f"{len(both_rated):,}")

    if len(both_rated) > 0:
        mean_abs_gap = both_rated["abs_gap"].mean()
        col3.metric("Mean |Gap|", f"{mean_abs_gap:.1f} pp")

        max_gap_row = both_rated.loc[both_rated["abs_gap"].idxmax()]
        col4.metric(
            "Largest Gap",
            f"{max_gap_row['abs_gap']:.0f} pp",
            delta=max_gap_row["display_label"],
            delta_color="off",
        )
    else:
        col3.metric("Mean |Gap|", "N/A")
        col4.metric("Largest Gap", "N/A")

    st.divider()

    # Main scatter plot
    col_scatter, col_controls = st.columns([4, 1])
    with col_controls:
        show_quadrants = st.checkbox("Show quadrant labels", value=True)
        st.caption(
            "Each point = one movie.\n\n"
            "**Above** diagonal → Critics rate higher.\n\n"
            "**Below** diagonal → Audiences rate higher."
        )

    with col_scatter:
        fig_scatter = create_scatter_plot(
            filtered_df,
            show_quadrants=show_quadrants,
            selected_movies=st.session_state.comparison_list,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="main_scatter")

    st.divider()

    # Gap ranking and distribution side by side
    col_rank, col_dist = st.columns(2)

    with col_rank:
        rank_mode = st.selectbox(
            "Ranking mode",
            options=["largest", "critics_higher", "audiences_higher"],
            format_func=lambda x: {
                "largest": "Largest Disagreement",
                "critics_higher": "Critics Rate Higher",
                "audiences_higher": "Audiences Rate Higher",
            }[x],
        )
        top_n = st.slider("Number of movies", 10, 50, 20, key="rank_topn")
        fig_rank = create_gap_ranking(filtered_df, mode=rank_mode, top_n=top_n)
        st.plotly_chart(fig_rank, use_container_width=True)

    with col_dist:
        fig_dist = create_gap_distribution(filtered_df)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Quick stats
        if len(both_rated) > 0:
            critics_higher_pct = (both_rated["rating_gap"] > 0).mean() * 100
            audiences_higher_pct = (both_rated["rating_gap"] < 0).mean() * 100
            agree_pct = (both_rated["rating_gap"] == 0).mean() * 100

            st.caption(
                f"🔴 Critics higher: **{critics_higher_pct:.1f}%** of movies · "
                f"🔵 Audiences higher: **{audiences_higher_pct:.1f}%** · "
                f"⚪ Equal: **{agree_pct:.1f}%**"
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2: MOVIE DETAIL
# ═══════════════════════════════════════════════════════════════
with tab_detail:
    # Movie selector
    st.subheader("Select a Movie")

    if search_query:
        search_results = filtered_df
    else:
        search_results = movies_df

    if len(search_results) == 0:
        st.warning("This movie is not included in the loaded dataset.")
    else:
        movie_options = search_results.sort_values("display_label")["display_label"].tolist()
        selected_label = st.selectbox(
            "Choose a movie",
            options=movie_options,
            index=0 if movie_options else None,
            key="movie_selector",
        )

        if selected_label:
            movie_row = search_results[search_results["display_label"] == selected_label].iloc[0]
            movie_id = movie_row["movie_id"]

            # Add to comparison button
            if movie_id not in st.session_state.comparison_list:
                if st.button(f"➕ Add to comparison", key="add_compare"):
                    st.session_state.comparison_list.append(movie_id)
                    st.rerun()
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
                st.subheader("Ratings")

                r_col1, r_col2 = st.columns(2)
                r_col1.metric("🍅 Critic", format_rating(movie_row.get("critic_rating")))
                r_col2.metric("🍿 Audience", format_rating(movie_row.get("audience_rating")))

                gap_val = movie_row.get("rating_gap")
                st.metric(
                    f"{gap_color(gap_val)} Rating Gap",
                    format_gap(gap_val),
                    help="Critic Rating − Audience Rating (percentage points)"
                )

                c_col1, c_col2 = st.columns(2)
                c_col1.caption(f"Critic count: {format_count(movie_row.get('critic_count'))}")
                c_col2.caption(f"Audience count: {format_count(movie_row.get('audience_count'))}")

            st.divider()

            # --- Critic Review Sentiment ---
            st.subheader("📝 Critic Review Text Sentiment (VADER)")
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
                    pct_pos = movie_row.get("pct_positive", 0) or 0
                    pct_neu = movie_row.get("pct_neutral", 0) or 0
                    pct_neg = movie_row.get("pct_negative", 0) or 0
                    n_analyzed = int(movie_row.get("reviews_analyzed", 0) or 0)

                    fig_sent = create_sentiment_chart(
                        pct_pos, pct_neu, pct_neg, n_analyzed,
                        title=f"Critic Review Sentiment — {movie_row.get('title', '')}"
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)

                with col_sent_stats:
                    avg_sent = movie_row.get("avg_sentiment")
                    st.metric(
                        "Avg. Compound Score",
                        f"{avg_sent:.3f}" if pd.notna(avg_sent) else "N/A",
                    )
                    st.metric("Reviews Analyzed", f"{n_analyzed:,}")
                    st.metric("Positive", f"{pct_pos:.1f}%")
                    st.metric("Neutral", f"{pct_neu:.1f}%")
                    st.metric("Negative", f"{pct_neg:.1f}%")

                st.divider()

                # --- Review Explorer ---
                st.subheader("📖 Review Explorer")

                # Review filters
                rev_col1, rev_col2, rev_col3 = st.columns(3)
                with rev_col1:
                    sent_filter = st.multiselect(
                        "Sentiment",
                        options=["Positive", "Neutral", "Negative"],
                        default=["Positive", "Neutral", "Negative"],
                        key="sent_filter",
                    )
                with rev_col2:
                    pubs = movie_reviews["publication"].dropna().unique().tolist()
                    pub_filter = st.multiselect(
                        "Publication", options=sorted(pubs), default=[], key="pub_filter"
                    )
                with rev_col3:
                    rev_type_filter = st.multiselect(
                        "Review Type",
                        options=["Fresh", "Rotten"],
                        default=["Fresh", "Rotten"],
                        key="rev_type_filter",
                    )

                # Apply review filters
                display_reviews = movie_reviews.copy()
                if sent_filter:
                    display_reviews = display_reviews[
                        display_reviews["sentiment_label"].isin(sent_filter)
                    ]
                if pub_filter:
                    display_reviews = display_reviews[
                        display_reviews["publication"].isin(pub_filter)
                    ]
                if rev_type_filter:
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
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Review": st.column_config.TextColumn(width="large"),
                            "Score": st.column_config.NumberColumn(format="%.3f"),
                        },
                    )
                else:
                    st.info("No reviews match the current filters.")


# ═══════════════════════════════════════════════════════════════
# TAB 3: COMPARE
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("⚖️ Movie Comparison")

    if not st.session_state.comparison_list:
        st.info(
            "No movies in your comparison list yet.\n\n"
            "Go to the **Movie Detail** tab, search for movies, and click "
            "**➕ Add to comparison** to build your list."
        )
    else:
        comp_df = movies_df[movies_df["movie_id"].isin(st.session_state.comparison_list)]

        if len(comp_df) == 0:
            st.warning("Selected movies not found in dataset.")
        else:
            # Comparison table
            st.markdown("### Rating Comparison")
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
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Critic %": st.column_config.NumberColumn(format="%.0f%%"),
                    "Audience %": st.column_config.NumberColumn(format="%.0f%%"),
                    "Gap (pp)": st.column_config.NumberColumn(format="%+.0f"),
                    "Critic Count": st.column_config.NumberColumn(format="%,.0f"),
                    "Audience Count": st.column_config.NumberColumn(format="%,.0f"),
                },
            )

            # Comparison bar chart
            fig_comp = create_comparison_bars(comp_df)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Sentiment comparison
            st.markdown("### Critic Review Sentiment Comparison")
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
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Avg. Sentiment": st.column_config.NumberColumn(format="%.3f"),
                    "% Positive": st.column_config.NumberColumn(format="%.1f%%"),
                    "% Neutral": st.column_config.NumberColumn(format="%.1f%%"),
                    "% Negative": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

            st.divider()

            # CSV Export
            st.markdown("### 📥 Export")
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
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════
# TAB 4: DATASET INFO
# ═══════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("ℹ️ About This Dataset")

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
    col1.metric("Total Movies", f"{total_movies:,}")
    col2.metric("With Both Ratings", f"{both_rated_total:,}")
    col3.metric("Total Critic Reviews", f"{total_reviews:,}")
    col4.metric("Movies with Reviews", f"{movies_with_reviews:,}")

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
    st.subheader("📊 Critic Text Sentiment vs. Critic Rating")
    st.caption(
        "Comparing VADER compound scores of critic review text against the aggregate "
        "Tomatometer rating. Points far from the diagonal suggest the text tone differs "
        "from the overall rating."
    )
    fig_svr = create_sentiment_vs_rating(movies_df)
    st.plotly_chart(fig_svr, use_container_width=True)

    st.divider()

    # Data quality overview
    st.subheader("📋 Data Quality Summary")

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
    st.dataframe(pd.DataFrame(quality_data), hide_index=True, use_container_width=True)

    # Full filtered export
    st.divider()
    st.subheader("📥 Export All Filtered Data")

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
        use_container_width=True,
    )
