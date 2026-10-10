# 🍅 The Rotten Tomatoes Gap

**Quantifying the Critic vs. Audience Disconnect**

A Streamlit dashboard that explores differences between critic and audience opinions using [Stefano Leone's Rotten Tomatoes Movies and Critic Reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) from Kaggle (CC0: Public Domain).

## Features

- **Search & Browse**: Search all 17k+ movie titles, filter by genre, year, and rating counts; the same filters carry into Movie Detail
- **Scatter Plot**: Interactive critic vs. audience rating visualization with quadrant labels
- **Gap Ranking**: Rank movies by largest disagreement, critics higher, or audiences higher
- **Critic Sentiment**: VADER analysis of critic review text with per-movie breakdowns
- **Review Explorer**: Browse individual critic reviews with sentiment scores
- **Compare Movies**: Side-by-side comparison of up to five movies; manage the shortlist in the Compare tab
- **In-app explanations**: Hover over circled info icons to understand filters, scores, charts, and sentiment measures
- **CSV Export**: Download filtered results

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Both PyArrow and Fastparquet are included as Parquet readers. Pandas can use
Fastparquet if a local policy blocks PyArrow's native DLLs.

### 2. Preprocess the data

The preprocessing script will automatically download the dataset via KaggleHub:

```bash
python preprocess.py
```

If KaggleHub fails, download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) and extract the CSVs into `data/raw/`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── preprocess.py           # One-time data preprocessing CLI
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dashboard theme config
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning & validation
│   ├── sentiment.py        # VADER sentiment scoring
│   ├── data_loader.py      # Cached data loading
│   └── charts.py           # Plotly chart builders
├── data/
│   ├── raw/                # Place downloaded CSVs here (if not using KaggleHub)
│   └── processed/          # Generated Parquet files
└── README.md
```

## Dataset

- **Source**: Stefano Leone / Kaggle
- **License**: CC0: Public Domain
- **Snapshot date**: October 31, 2020
- **Coverage**: ~17,000+ movies with critic and audience ratings
- **Reviews**: Individual critic review text (no audience text reviews)

## Key Metrics

| Metric | Formula | Meaning |
|:--|:--|:--|
| Rating Gap | Critic % − Audience % | Direction of disagreement |
| Disagreement Size | \|Critic % − Audience %\| | Magnitude of disagreement |

## Limitations

- Historical snapshot, not live scores
- VADER may misinterpret sarcasm
- No audience review text available
- Rating percentages reflect different reviewer populations
