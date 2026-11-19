# Research design

The full, approved methodology (20 issues, decision table, figures plan) lives in the team doc:

**[S&P 500 Co-movement Networks — Research Design](https://claude.ai/artifact/1vSMkis2Yrw7ZaxBpx4c2z)** (approved 19 Nov 2026)

Every numeric parameter from that document is in [`../config.yaml`](../config.yaml), each key tagged with its issue number.

## Decisions at a glance

| # | Decision |
| --- | --- |
| 1 | Fixed universe: end-2024 constituents with full 2015–2024 history |
| 2 | End-2024 GICS labels; 2018/2023 reclassified stocks excluded as robustness |
| 3 | One share class per issuer (longest history, then dollar volume) |
| 4 | yfinance primary; Tiingo REST API validates 30 tickers + SPY on log returns |
| 5 | MediaWiki API, pinned revision, parsed with BeautifulSoup |
| 6 | No imputation; ≥99% coverage, no gap > 5 days; pairwise-complete correlations |
| 7 | SPY market model per window; raw returns for connectivity, residuals for communities |
| 8 | Noise handled by testing + MST; Marchenko–Pastur diagnostic reported |
| 9 | Fisher z + Bonferroni on residual returns (BH and fixed density as robustness) |
| 10 | Positive edges only, weight ρ; distance √(2(1−ρ)) for paths and MST |
| 11 | Raw matrix/MST for connectivity; residual network for communities and GICS |
| 12 | Strength, MST degree and betweenness; PageRank dropped (redundancy check only) |
| 13 | Louvain, 50 seeds, γ = 1; NMI/ARI reported as median + IQR |
| 14 | Arithmetic NMI and ARI, implemented from scratch and checked with sklearn |
| 15 | Label-permutation null (main); time shuffle and rewiring on snapshots |
| 16 | Rolling 2-year windows + calendar-year snapshots (2019, 2020, 2022) + 63-day layer |
| 17 | Top-100 by 2015 dollar volume |
| 18 | Rolling series descriptive; formal tests on snapshots only |
| 19 | Static figures; Looker Studio optional, decided later |
| 20 | Confirm deadlines with professors |
