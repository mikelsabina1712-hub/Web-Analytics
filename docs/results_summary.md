# Results summary

Final figures and tables: Drive `WebAnalytics_SP500/outputs/report/` (zip: `outputs/report_assets.zip`), produced by Notebook 08.

## Data
- Constituents: Wikipedia *List of S&P 500 companies* via the MediaWiki API, revision 1265285344 (26 Dec 2024): 503 securities, 500 issuers.
- Prices: Yahoo Finance via yfinance (489 series); Tiingo REST API for the 15 tickers Yahoo no longer serves. Yahoo vs Tiingo validation: 29/31 passed; KDP (2018-07-10) and RTX (2020-04-03) are corporate-action days, set to missing.
- Ticker changes after 2024 handled in `config.yaml`: BK→BNY, MMC→MRSH, EQR→VMRK.
- Fixed universe: **468 stocks**, 2015-01-02 to 2024-12-31 (2,516 days). Excluded: 31 late listings (incl. PARA, whose pre-2021 history neither free source keeps), 2 stale/gap series (AMCR, SW), 2 second share classes. Missing stock-days: 0.001%.
- GICS reclassifications: 23 stocks into Communication Services in 2018 and 14 changes in 2023, both matching published counts; FOX/FOXA excluded as a different company (21st Century Fox vs Fox Corp). In the universe: 16 (2018) and 13 (2023).

## Main results (residual returns after the SPY market model; Pearson; Bonferroni)

| | 2019 | 2020 | 2022 |
|---|---|---|---|
| Mean raw correlation | 0.25 | 0.54 | 0.42 |
| Residual network density | 3.4% | 11.2% | 4.5% |
| NMI vs GICS (median of 50 seeds) | 0.58 | 0.24 | 0.54 |
| ARI vs GICS | 0.39 | 0.10 | 0.32 |
| Modularity of the GICS partition | 0.47 | 0.18 | 0.51 |
| Louvain communities | 20 | 9 | 15 |

- The GICS match beats all 1,000 label permutations in all 100 windows (z 34–155).
- **COVID:** connectivity jumps (two-year mean correlation 0.30 → 0.49; 63-day peak 0.70 in the window ending 18 Mar 2020) while GICS recovery collapses (NMI ≈ 0.61 → 0.27). Beyond-market structure gets denser but cross-sector: defensives (utilities core), credit and reopening (regional banks, consumer lenders, malls, energy), insurers with large banks and industrials, software and chips.
- **Calm periods:** communities follow industries and themes more than sectors (semiconductors, regional banks, medtech, bond-like defensives, travel, housing), which caps NMI near 0.6.
- Best-recovered sectors: Energy, Utilities, Health Care, Real Estate. Worst: Communication Services, Consumer Discretionary, IT (split).
- Raw-MST hubs: large banks and MSFT (2019), diversified industrials (2020, 2022).

## Validation and robustness
- Degree-preserving rewiring: observed Louvain Q far above all rewired networks (z 193–229).
- Time-shuffle null: Pearson with Bonferroni admits a few false edges under fat tails (at most 0.4% of edges); Spearman is calibrated (2 false edges in 60 networks). Decision: Pearson stays the main specification, Spearman is a full robustness run.
- COVID NMI drop under every specification: main 0.34, Spearman 0.30, equal-weighted leave-one-out market 0.31, BH 0.29, top 100 0.20, fixed density 0.17, excluding reclassified stocks 0.36. 2020 is the lowest year under all 14 specifications.
- Caveats: fixed density halves the drop (part of it is density-driven); Spearman makes it gradual instead of a cliff (extreme days drive the abruptness); the late-sample decline partly reflects SPY's mega-cap concentration (the equal-weighted proxy recovers to ≈ 0.6 by 2024); communities on raw returns barely match GICS (removing the market is essential); raw and residual MST hubs share only 1–2 of their top 10.
