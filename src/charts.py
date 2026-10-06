"""
Plotly chart builder module for the Rotten Tomatoes Gap dashboard.

All charts are built with Plotly Express / Graph Objects for
interactivity and consistent styling.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# --- Color constants ---
COLOR_CRITICS_HIGHER = "#E63946"   # Red — critics rate higher
COLOR_AUDIENCES_HIGHER = "#457B9D" # Blue — audiences rate higher
COLOR_NEUTRAL = "#A8DADC"          # Light teal — agreement
COLOR_POSITIVE = "#2A9D8F"         # Green for positive sentiment
COLOR_NEGATIVE = "#E76F51"         # Orange-red for negative
COLOR_NEUTRAL_SENT = "#E9C46A"     # Yellow for neutral sentiment

DIVERGING_COLORSCALE = [
    [0.0, "#457B9D"],   # Audiences higher (negative gap)
    [0.5, "#F1FAEE"],   # Agreement (zero gap)
    [1.0, "#E63946"],   # Critics higher (positive gap)
]


def create_scatter_plot(
    df: pd.DataFrame,
    show_quadrants: bool = True,
    height: int = 600,
    selected_movies: list[str] | None = None,
) -> go.Figure:
    """
    Create the main critic vs. audience rating scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Movies DataFrame with critic_rating, audience_rating, rating_gap.
    show_quadrants : bool
        Whether to show the 60% reference lines and quadrant labels.
    height : int
        Chart height in pixels.
    selected_movies : list[str] or None
        Movie IDs to highlight on the plot.

    Returns
    -------
    go.Figure
    """
    # Filter to movies with both ratings
    plot_df = df[df["has_both_ratings"]].copy()

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No movies with both ratings in current filter",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    # Normalize gap for color mapping: map [-100, 100] → [0, 1]
    plot_df["gap_normalized"] = (plot_df["rating_gap"] + 100) / 200

    fig = go.Figure()

    # Main scatter points
    fig.add_trace(go.Scatter(
        x=plot_df["audience_rating"],
        y=plot_df["critic_rating"],
        mode="markers",
        marker=dict(
            size=6,
            color=plot_df["rating_gap"],
            colorscale=DIVERGING_COLORSCALE,
            cmin=-100,
            cmax=100,
            colorbar=dict(
                title="Rating Gap<br>(C − A)",
                tickvals=[-100, -50, 0, 50, 100],
                ticktext=["-100<br>Audiences<br>higher", "-50", "0<br>Equal", "+50", "+100<br>Critics<br>higher"],
                len=0.8,
            ),
            opacity=0.6,
            line=dict(width=0.5, color="white"),
        ),
        text=plot_df["display_label"],
        customdata=np.column_stack([
            plot_df["critic_rating"],
            plot_df["audience_rating"],
            plot_df["rating_gap"],
            plot_df.get("critic_count", pd.Series(dtype="float64")),
            plot_df.get("audience_count", pd.Series(dtype="float64")),
        ]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Critic Rating: %{customdata[0]:.0f}%<br>"
            "Audience Rating: %{customdata[1]:.0f}%<br>"
            "Gap: %{customdata[2]:+.0f} pp<br>"
            "Critic Count: %{customdata[3]:,.0f}<br>"
            "Audience Count: %{customdata[4]:,.0f}<br>"
            "<extra></extra>"
        ),
        name="Movies",
    ))

    # Highlight selected movies
    if selected_movies:
        sel_df = plot_df[plot_df["movie_id"].isin(selected_movies)]
        if len(sel_df) > 0:
            fig.add_trace(go.Scatter(
                x=sel_df["audience_rating"],
                y=sel_df["critic_rating"],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=COLOR_CRITICS_HIGHER,
                    symbol="star",
                    line=dict(width=2, color="black"),
                ),
                text=sel_df["display_label"],
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo="skip",
                name="Selected",
            ))

    # Diagonal line (equal ratings)
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        line=dict(color="gray", dash="dash", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Optional quadrant lines at 60%
    if show_quadrants:
        for val in [60]:
            fig.add_hline(y=val, line_dash="dot", line_color="lightgray",
                         line_width=1, opacity=0.7)
            fig.add_vline(x=val, line_dash="dot", line_color="lightgray",
                         line_width=1, opacity=0.7)

        # Quadrant labels
        annotations = [
            dict(x=30, y=80, text="Critics Prefer", font=dict(color="rgba(230,57,70,0.5)", size=12)),
            dict(x=80, y=30, text="Crowd Pleasers", font=dict(color="rgba(69,123,157,0.5)", size=12)),
            dict(x=80, y=80, text="Both Positive", font=dict(color="rgba(42,157,143,0.5)", size=12)),
            dict(x=30, y=30, text="Both Negative", font=dict(color="rgba(150,150,150,0.5)", size=12)),
        ]
        for ann in annotations:
            fig.add_annotation(
                x=ann["x"], y=ann["y"], text=ann["text"],
                showarrow=False, font=ann["font"],
                opacity=0.7,
            )

    fig.update_layout(
        title="Critic vs. Audience Ratings",
        xaxis=dict(title="Audience Rating (%)", range=[-2, 102], dtick=20),
        yaxis=dict(title="Critic Rating (%)", range=[-2, 102], dtick=20),
        height=height,
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_gap_ranking(
    df: pd.DataFrame,
    mode: str = "largest",
    top_n: int = 20,
    height: int = 500,
) -> go.Figure:
    """
    Create a horizontal bar chart ranking movies by rating gap.

    Parameters
    ----------
    df : pd.DataFrame
        Movies DataFrame with rating_gap and abs_gap.
    mode : str
        'largest' for biggest |gap|, 'critics_higher' for positive gaps,
        'audiences_higher' for negative gaps.
    top_n : int
        Number of movies to show.
    height : int
        Chart height.

    Returns
    -------
    go.Figure
    """
    plot_df = df[df["has_both_ratings"]].copy()

    if mode == "largest":
        plot_df = plot_df.nlargest(top_n, "abs_gap")
        sort_col = "abs_gap"
        title = f"Top {top_n} Largest Rating Disagreements"
    elif mode == "critics_higher":
        plot_df = plot_df[plot_df["rating_gap"] > 0].nlargest(top_n, "rating_gap")
        sort_col = "rating_gap"
        title = f"Top {top_n} Movies Where Critics Rate Higher"
    elif mode == "audiences_higher":
        plot_df = plot_df[plot_df["rating_gap"] < 0].nsmallest(top_n, "rating_gap")
        sort_col = "rating_gap"
        title = f"Top {top_n} Movies Where Audiences Rate Higher"
    else:
        plot_df = plot_df.nlargest(top_n, "abs_gap")
        sort_col = "abs_gap"
        title = f"Top {top_n} Largest Rating Disagreements"

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No movies match this filter",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False)
        return fig

    plot_df = plot_df.sort_values("rating_gap", ascending=True)

    colors = [COLOR_CRITICS_HIGHER if g > 0 else COLOR_AUDIENCES_HIGHER
              for g in plot_df["rating_gap"]]

    fig = go.Figure(go.Bar(
        x=plot_df["rating_gap"],
        y=plot_df["display_label"],
        orientation="h",
        marker=dict(color=colors),
        customdata=np.column_stack([
            plot_df["critic_rating"],
            plot_df["audience_rating"],
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Gap: %{x:+.0f} pp<br>"
            "Critic: %{customdata[0]:.0f}% | Audience: %{customdata[1]:.0f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color="gray", line_width=1)

    fig.update_layout(
        title=title,
        xaxis=dict(title="Rating Gap (Critic − Audience, pp)", range=[-105, 105]),
        yaxis=dict(title=""),
        height=max(height, top_n * 28),
        template="plotly_white",
        showlegend=False,
    )

    return fig


def create_sentiment_chart(
    pct_positive: float,
    pct_neutral: float,
    pct_negative: float,
    reviews_analyzed: int,
    title: str = "Critic Review Sentiment Distribution",
) -> go.Figure:
    """
    Create a donut chart showing sentiment distribution for a movie's critic reviews.

    Parameters
    ----------
    pct_positive, pct_neutral, pct_negative : float
        Percentage of reviews in each category.
    reviews_analyzed : int
        Total number of reviews scored.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    labels = ["Positive", "Neutral", "Negative"]
    values = [pct_positive, pct_neutral, pct_negative]
    colors = [COLOR_POSITIVE, COLOR_NEUTRAL_SENT, COLOR_NEGATIVE]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        annotations=[dict(
            text=f"{reviews_analyzed}<br>reviews",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False,
        )],
        height=350,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    return fig


def create_sentiment_vs_rating(
    df: pd.DataFrame,
    height: int = 500,
) -> go.Figure:
    """
    Scatter plot comparing average VADER compound score to critic rating.

    Parameters
    ----------
    df : pd.DataFrame
        Merged movies+sentiment DataFrame.
    height : int
        Chart height.

    Returns
    -------
    go.Figure
    """
    plot_df = df.dropna(subset=["avg_sentiment", "critic_rating"]).copy()

    if len(plot_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    fig = px.scatter(
        plot_df,
        x="avg_sentiment",
        y="critic_rating",
        hover_name="display_label",
        hover_data={
            "avg_sentiment": ":.3f",
            "critic_rating": ":.0f",
            "reviews_analyzed": True,
        },
        opacity=0.5,
        color_discrete_sequence=[COLOR_CRITICS_HIGHER],
    )

    fig.update_layout(
        title="Critic Text Sentiment vs. Critic Rating",
        xaxis=dict(title="Average VADER Compound Score", range=[-1.05, 1.05]),
        yaxis=dict(title="Critic Rating (%)", range=[-2, 102]),
        height=height,
        template="plotly_white",
    )

    return fig


def create_comparison_bars(
    df: pd.DataFrame,
    height: int = 400,
) -> go.Figure:
    """
    Grouped bar chart comparing critic and audience ratings for selected movies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of selected movies to compare.
    height : int
        Chart height.

    Returns
    -------
    go.Figure
    """
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No movies selected for comparison",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False)
        return fig

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Critic Rating",
        x=df["display_label"],
        y=df["critic_rating"],
        marker_color=COLOR_CRITICS_HIGHER,
        hovertemplate="<b>%{x}</b><br>Critic: %{y:.0f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Audience Rating",
        x=df["display_label"],
        y=df["audience_rating"],
        marker_color=COLOR_AUDIENCES_HIGHER,
        hovertemplate="<b>%{x}</b><br>Audience: %{y:.0f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Critic vs. Audience Rating Comparison",
        xaxis=dict(title=""),
        yaxis=dict(title="Rating (%)", range=[0, 105]),
        barmode="group",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_gap_distribution(
    df: pd.DataFrame,
    height: int = 400,
) -> go.Figure:
    """
    Histogram showing the distribution of rating gaps across all movies.

    Parameters
    ----------
    df : pd.DataFrame
        Movies DataFrame with rating_gap.
    height : int
        Chart height.

    Returns
    -------
    go.Figure
    """
    plot_df = df[df["has_both_ratings"]].copy()

    fig = px.histogram(
        plot_df,
        x="rating_gap",
        nbins=50,
        color_discrete_sequence=["#264653"],
        labels={"rating_gap": "Rating Gap (Critic − Audience, pp)"},
    )

    fig.add_vline(x=0, line_color="red", line_dash="dash", line_width=2,
                  annotation_text="Equal", annotation_position="top")

    mean_gap = plot_df["rating_gap"].mean()
    fig.add_vline(x=mean_gap, line_color="orange", line_dash="dot", line_width=2,
                  annotation_text=f"Mean: {mean_gap:+.1f}", annotation_position="bottom")

    fig.update_layout(
        title="Distribution of Rating Gaps",
        yaxis=dict(title="Number of Movies"),
        height=height,
        template="plotly_white",
        showlegend=False,
    )

    return fig
