# S&P 500 co-movement networks

UC3M · Web Analytics · Final project (Data Science & Engineering, 4th year)

**Research question.** Does a network built from the co-movement of S&P 500 stocks recover the official GICS sectors, and how does its structure change between calm (2019) and crisis periods (2020, 2022)?

The approved methodology is summarised in [`docs/research_design.md`](docs/research_design.md), with a link to the full design doc.

## Repository layout

```
notebooks/     01 … 08, run in order (each reads the previous one's files on Drive)
src/sp500net/  shared code imported by every notebook
config.yaml    every date, threshold and seed (one place to change them)
tests/         offline unit tests for src/ (`pytest`)
docs/          research design summary
data/          never committed; in Colab the data lives on Google Drive
```

| Notebook | What it does | Runtime | Open |
| --- | --- | --- | --- |
| 01 | Constituents (Wikipedia API), prices (Yahoo, Tiingo fallback), sanity checks, cross-source validation | ~10 min (first run) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/01_data_collection.ipynb) |
| 02 | Log returns, stale prices, coverage rule, share classes: fixed universe of 468 stocks | < 1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/02_returns_universe.ipynb) |
| 03 | 97 rolling, 3 snapshot and 491 event windows; SPY market model per window | ~1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/03_market_model.ipynb) |
| 04 | Correlations, Bonferroni residual networks, MSTs, connectivity and Marchenko–Pastur | ~2 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/04_networks.ipynb) |
| 05 | Centrality, Louvain (50 seeds), NMI/ARI/Q_GICS, label-permutation null | 10–20 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/05_communities.ipynb) |
| 06 | Time-shuffle and rewiring nulls; what the snapshot communities are | ~5 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/06_nulls.ipynb) |
| 07 | BH, fixed density, EW market, Spearman, top 100, reclassified, γ, raw, universe | 20–30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/07_robustness.ipynb) |
| 08 | Final figures, tables and the Looker Studio CSV | ~1 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/08_report_figures.ipynb) |

Long steps (Louvain in 05, robustness in 07) save each window to Drive as soon as it finishes: if Colab disconnects, *Run all* again and it continues.

**Outputs.** Intermediate files go to `MyDrive/WebAnalytics_SP500/data/`; tables and figures to `MyDrive/WebAnalytics_SP500/outputs/`. Notebook 08 collects the final figures and tables in `outputs/report/` and zips them as `outputs/report_assets.zip`.

## One-time setup (each team member)

1. **Colab ↔ GitHub.** In Colab, *File → Open notebook → GitHub*, tick *Include private repos*, and authorise. This lets you open and save notebooks from this repo.
2. **Colab Secrets** (key icon in Colab's left sidebar). Add:
   - `TIINGO_API_KEY`: your key from tiingo.com (free plan).
   - `GITHUB_TOKEN`: only while the repo is private. A fine-grained token with *read-only Contents* access to this repo, so the setup cell can clone `src/`.

   Toggle *Notebook access* on for both. Never paste a key into a cell.
3. **Wikipedia contact.** Put a team e-mail in `config.yaml → wikipedia.user_agent` (Wikimedia asks for one). Commit that change once.

Data is written to `MyDrive/WebAnalytics_SP500/data`. To share one copy across the team, one member creates that folder and shares it; the others add a shortcut to it in their own *My Drive*.

## Working together

- One notebook per pull request; at least one teammate reviews before merging.
- Change parameters only in `config.yaml`, never inside a notebook.
- Run `pytest` (or the test cell) after changing anything in `src/`.
- Before committing a notebook, *Edit → Clear all outputs* unless the outputs are needed for review, to keep diffs readable.

## Data sources and terms

- **Wikipedia**, *List of S&P 500 companies*, pinned revisions via the MediaWiki API (CC BY-SA 4.0).
- **Yahoo Finance** via the open-source `yfinance` library (unofficial; for research and education).
- **Tiingo** end-of-day REST API, free plan: prices for the 15 tickers Yahoo no longer serves, and cross-validation of a random sample of Yahoo series.

Price data is stored on Drive only and is not redistributed in this repository.
