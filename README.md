# S&P 500 co-movement networks

UC3M · Web Analytics · Final project (Data Science & Engineering, 4th year)

**Research question.** Does a network built from the co-movement of S&P 500 stocks recover the official GICS sectors, and how does its structure change between calm (2019) and crisis periods (2020, 2022)?

The approved methodology is summarised in [`docs/research_design.md`](docs/research_design.md), with a link to the full design doc.

## Repository layout

```
notebooks/   01_data_collection … 08_figures, run in order
src/sp500net/  shared code imported by every notebook
config.yaml  every date, threshold and seed (one place to change them)
tests/       offline unit tests for src/
outputs/     figures and small summary tables (committed)
docs/        research design summary
data/        never committed; in Colab the data lives on Google Drive
```

| Notebook | Status | Open |
| --- | --- | --- |
| 01 Data collection | Ready to run | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikelsabina1712-hub/web-analytics/blob/main/notebooks/01_data_collection.ipynb) |
| 02 Returns and universe | Next | |
| 03 Market model | Planned | |
| 04 Network construction | Planned | |
| 05 Metrics and communities | Planned | |
| 06 Validation (null models) | Planned | |
| 07 Robustness | Planned | |
| 08 Figures and tables | Planned | |

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
- **Tiingo** end-of-day API, free plan, used only to validate a sample of prices.

Price data is stored on Drive only and is not redistributed in this repository.
