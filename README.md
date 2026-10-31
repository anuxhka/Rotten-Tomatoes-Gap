# 🍅 The Rotten Tomatoes Gap

**Quantifying the Critic vs. Audience Disconnect**

An interactive Streamlit dashboard that measures how often, how much and in which direction film critics and audiences disagree. It covers 17,712 films and about 952,000 critic reviews from [Stefano Leone's Rotten Tomatoes Movies and Critic Reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) (Kaggle, CC0: Public Domain).

> **Data disclosure:** This is an independent analysis of a third-party dataset scraped on 31 October 2020. It does not use live Rotten Tomatoes data and is not affiliated with Rotten Tomatoes.

## Features

- **Search and filters**: search all titles (with match-case and exact-word options) and filter by genre, release year and rating counts. Filters apply across every tab.
- **Opinion map**: an interactive scatter plot of critic vs. audience scores with quadrant labels.
- **Disagreement leaderboard**: rank films by largest disagreement, critics rating higher or audiences rating higher.
- **Gap distribution**: a histogram of rating gaps across the filtered films.
- **Critic sentiment**: VADER sentiment analysis of critic review text, with a per-film breakdown.
- **Review explorer**: browse individual critic reviews filtered by sentiment, publication and Fresh/Rotten verdict.
- **Compare**: compare up to five films side by side.
- **CSV export**: download the filtered films or the comparison shortlist.

## Tech Stack

Python · pandas · Streamlit · Plotly · VADER (vaderSentiment) · Parquet (PyArrow / fastparquet)

## Getting Started

### Prerequisites

- Python 3.12 or newer

### Installation

```bash
git clone https://github.com/anuxhka/Rotten-Tomatoes-Gap.git
cd Rotten-Tomatoes-Gap
pip install -r requirements.txt
```

### Running the dashboard

```bash
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. The processed data is included in `data/processed/`, so no download is needed to run it.

## Rebuilding the Data

The processed Parquet files can be regenerated from the raw dataset:

```bash
pip install -r requirements-preprocess.txt
python preprocess.py
```

The script downloads the dataset with KaggleHub. Alternatively, download it manually from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) and extract the CSV files into `data/raw/`.

Preprocessing cleans and validates the ratings, computes the rating gap for each film, removes empty and duplicate reviews, and scores every review with VADER.

## Testing

```bash
python -B -m unittest discover -s tests -v
```

The suite covers data cleaning, sentiment scoring, filters, charts, the integrity of the bundled data, and headless end-to-end runs of the dashboard using Streamlit's `AppTest`.

## Deployment

The project includes an ASGI entry point (`api/index.py`) and a `vercel.json` configuration for deployment on [Vercel](https://vercel.com/docs/cli/deploying-from-cli):

```bash
vercel deploy          # preview deployment
vercel deploy --prod   # production deployment
```

The ASGI app can be run locally with `python -m uvicorn api.index:app --port 8502`.

Note that the bundled data is about 100 MB, and Vercel limits how long a WebSocket session can stay open, so long dashboard sessions may reconnect and reset the comparison shortlist.

## Project Structure

```
├── app.py                       # Streamlit dashboard
├── preprocess.py                # Data preprocessing pipeline
├── api/
│   └── index.py                 # ASGI entry point for Vercel
├── src/
│   ├── preprocessing.py         # Data cleaning and validation
│   ├── sentiment.py             # VADER sentiment scoring
│   ├── data_loader.py           # Cached data loading
│   ├── filters.py               # Title search
│   └── charts.py                # Plotly chart builders
├── data/
│   ├── raw/                     # Raw CSV files (optional, for preprocessing)
│   └── processed/               # Processed Parquet files
├── tests/                       # Unit and end-to-end tests
├── assets/styles.css            # Dashboard styling
├── .streamlit/config.toml       # Streamlit theme
├── report/                      # Mini project report
├── requirements.txt             # Dashboard dependencies
├── requirements-preprocess.txt  # Preprocessing dependencies
└── vercel.json                  # Vercel configuration
```

## Dataset

| | |
|:--|:--|
| Source | Stefano Leone, Kaggle |
| License | CC0: Public Domain |
| Snapshot date | 31 October 2020 |
| Films | 17,712 (17,407 with both critic and audience scores) |
| Critic reviews | ~952,000 with review text |
| Audience data | Aggregate scores only (no review text) |

## Key Metrics

| Metric | Formula | Meaning |
|:--|:--|:--|
| Rating gap | Critic % − Audience % | Who liked the film more: positive means critics, negative means audiences |
| Absolute gap | \|Critic % − Audience %\| | How much they disagree, regardless of direction |
| Review sentiment | VADER compound score (−1 to +1) | Tone of the critic review text: Positive ≥ 0.05, Negative ≤ −0.05, otherwise Neutral |

## Limitations

- The data is a historical snapshot, not live scores.
- VADER can misread sarcasm and mixed opinions.
- No audience review text is available, so sentiment covers critics only.
- Critic and audience scores come from different groups and are measured differently.

## Authors

- Anushka Sinha (245816322)
- Aahna Negi (245816190)

## Acknowledgements

- Stefano Leone for the [Rotten Tomatoes Movies and Critic Reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- C. J. Hutto and E. Gilbert for [VADER](https://github.com/cjhutto/vaderSentiment)
