# 🍅 The Rotten Tomatoes Gap

**Quantifying the Critic vs. Audience Disconnect**

A Streamlit dashboard that explores differences between critic and audience opinions using [Stefano Leone's Rotten Tomatoes Movies and Critic Reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) from Kaggle (CC0: Public Domain).

**Data disclosure:** This is an independent analysis of a third-party dataset scraped on October 31, 2020. It does not use live Rotten Tomatoes website data and is not affiliated with Rotten Tomatoes. The same notice appears in the dashboard sidebar.

## Features

- **Search & Browse**: Search all 17k+ movie titles with match-case and exact-word options; filter by genre, year, and rating counts. The same filters carry into Movie Detail.
- **Scatter Plot**: Interactive critic vs. audience rating visualization with quadrant labels and responsive score-pair hover
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

Install the optional preprocessing dependencies before rebuilding the snapshot:

```bash
pip install -r requirements-preprocess.txt
```

The preprocessing script can then download the dataset via KaggleHub:

```bash
python preprocess.py
```

If KaggleHub fails, download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) and extract the CSVs into `data/raw/`.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

### 4. Run the tests

```bash
python -B -m unittest discover -s tests -v
```

## Vercel live demo (you deploy)

The repository includes an ASGI entry point in `api/index.py`, a catch-all
route in `vercel.json`, and a Python 3.12 version file. The processed Parquet
snapshot is bundled, so deployment does not download from Kaggle or need an
API key. The app and data disclosure can be checked locally with:

```bash
python -m uvicorn api.index:app --port 8502
```

Open `http://localhost:8502/` and check Overview, Movie Detail, and Compare.
When you are ready, use the [Vercel CLI](https://vercel.com/docs/cli/deploying-from-cli)
from this folder to create a **preview deployment first**, then a production
deployment after testing. The CLI will ask you to sign in to your own Vercel
account; no account credentials or tokens belong in this project:

```bash
npm install -g vercel
vercel login
vercel deploy
# After checking the preview URL:
vercel deploy --prod
```

This Vercel configuration is **locally tested via ASGI, but not verified on
Vercel**. Streamlit's ASGI support and Vercel's Python WebSocket support are
recent. On the Hobby plan, WebSocket connections are limited by the Function's
maximum duration and may reconnect after about five minutes; in-memory
session selections can reset. The processed snapshot is about 102 MB, so
check the preview build's Function bundle size and memory use before going live.

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── api/index.py            # ASGI entry point for Vercel
├── preprocess.py           # One-time data preprocessing CLI
├── requirements.txt        # Python dependencies
├── requirements-preprocess.txt # Optional data preparation dependencies
├── vercel.json             # Vercel route and Function configuration
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
