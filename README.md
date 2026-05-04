# Strategic Data Center Siting

**A forward-looking, risk-adjusted siting framework for hyperscale data centers across five U.S. markets.**

> *INDENG 290 LEC 005— Energy Analytics, UC Berkeley · Spring 2026*
>
> *Yijun Gu · Ruixin Wang · Jasmine Chen · David Zhao*

---

## TL;DR

We built a 10-year total-cost-of-ownership model for a 100 MW hyperscale facility across **Ashburn VA, Dallas TX, Phoenix AZ, Atlanta GA, and Chicago IL**, combining forward electricity prices, climate-driven cooling penalties, calibrated grid-stress probabilities, and Monte Carlo uncertainty over fuel, weather, and carbon. The framework ranks markets on both **expected cost** and **CVaR-95 risk-adjusted cost**, then stress-tests the ranking across six future-state scenarios.

**Headline finding:** Dallas wins on both metrics in every sensitivity case tested. Atlanta moves from #2 cheapest on nominal price to dead last when emissions intensity and capacity-scarcity premiums are priced in — confirming that **a nominal-price-only screen systematically misallocates hyperscale capital**.

| Market | Mean NPV (M$) | CVaR-95 NPV (M$) | Robust Rank | Verdict |
|---|---:|---:|---:|---|
| Dallas, TX | ~800 | ~841 | **#1** in 6/6 cases | ✅ Primary |
| Phoenix, AZ | ~880 | ~920 | #2/3 (gas-price dependent) | 🥈 Backup |
| Chicago, IL | ~895 | ~932 | #2/3 (gas-price dependent) | 🥈 Backup |
| Ashburn, VA | ~960 | ~1,022 | **#4** in 6/6 cases | ➖ Not recommended |
| Atlanta, GA | ~1,015 | ~1,090 | **#5** in 6/6 cases | ❌ Avoid |

*(Numbers are illustrative; precise values are produced by `notebooks/data_center_siting.ipynb` Section 10.)*

---

## What this project demonstrates

- **Forward-price construction from disparate forecasting products.** AEO 2026 sets the national trajectory; Cambium 2024 supplies regional levels and intra-year shape. The decomposition (`forward_price.py`) avoids the gating concern that historical prices alone cannot anchor a 10-year TCO.
- **Calibrated probabilistic modeling on imbalanced time-series data.** A multi-model classifier (LR / RF / XGBoost) with **time-blocked CV**, isotonic calibration, and Brier-score model selection — chosen over headline ROC-AUC because downstream cost calculations require well-calibrated probabilities, not just discriminative ones.
- **Monte Carlo TCO with CVaR-95 risk metric.** Stochastic gas prices, weather shocks, and discrete carbon scenarios over 5,000 runs × 5 markets × 10 years. Replaces ad-hoc volatility heuristics with a proper conditional-tail-expectation metric.
- **Robustness analysis.** Six sensitivity cases (gas spike, gas glut, climate warming, carbon-price shifts, demand growth) re-rank all five markets and quantify how stable each headline rank is. Includes a **break-even analysis** that solves for the subjective probability at which Phoenix and Chicago crossover under gas-price uncertainty.
- **Interconnection-queue caveat.** A market's energy-cost rank doesn't matter if you can't actually build there. Section 13 incorporates LBNL's *Queued Up 2025 Edition* queue-duration data per market as a post-hoc reality check on the recommendation.

---

## Methodology — four-component framework

```
forward_price[market, year, month, hour]
    = AEO_national[year]                        ← trajectory
      × regional_ratio[GEA(market), year]       ← levels
      × shape_factor[GEA(market), year, mo, h]  ← intra-year structure
```

| Layer | Source | Role |
|---|---|---|
| Trajectory | AEO 2026 Counterfactual Baseline (industrial nominal cents/kWh × 10) | Year-by-year national price level, 2026–2035 |
| Regional differentiation | Cambium 2024 MidCase annual end-use cost, load-weighted to GEA | Persistent geographic spread |
| Intra-year shape | Cambium 2024 month-hour, normalized so 288-cell mean = 1 per (gea, year) | Daily and seasonal structure |
| Climate adjustment | Open-Meteo historical temperatures + dynamic PUE function | Per-hour cooling penalty |
| Stress probability | XGBoost classifier on EIA-930 net-load × temperature features | Probabilistic stress-event premium |
| Uncertainty | Monte Carlo: gas σ=0.20 log-normal · temp σ=1.5°C normal · carbon $0/$25/$75 discrete | Risk-adjusted ranking |

---

## Repository structure

```
.
├── notebooks/
│   └── data_center_siting.ipynb     # Main analysis notebook (run this end-to-end)
├── src/
│   ├── config.py                    # Markets, mappings, modeling constants
│   ├── api_clients.py               # EIA-930 + Open-Meteo wrappers (cached to parquet)
│   ├── data_loader.py               # AEO + Cambium loaders
│   ├── historical_panel.py          # Hourly panel build + stress labeling
│   ├── forward_price.py             # AEO × Cambium hybrid forward-price builder
│   ├── stress_model.py              # Multi-model classifier with time-blocked CV
│   ├── stress_plots.py              # Plotting helpers
│   ├── cost_model.py                # Climate- and risk-adjusted hourly cost panel
│   ├── tco_simulation.py            # Monte Carlo TCO + CVaR
│   ├── sensitivity.py               # 6-case sensitivity sweep
│   └── final_ranking.py             # Robustness scorecard, break-even, recommendation
├── data/
│   └── raw/                         # All input datasets included — see data/raw/README.md
│       ├── Table_8._Electricity_Supply_Disposition_Prices_and_Emissions.csv
│       ├── Cambium24_*.csv          # 5 NREL Cambium scenario files
│       ├── eia930/*.parquet         # Cached EIA-930 hourly demand and fuel mix
│       └── weather/*.parquet        # Cached Open-Meteo hourly temperature
├── LICENSE                          # MIT
└── README.md
```

The repository ships with **all input data included**, so cloning and running the notebook produces the same results without any external downloads or API calls.

---

## Reproducing the analysis

### 1. Install Python dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost requests pyarrow nbformat jupyter
```

### 2. Run the notebook

```bash
jupyter lab notebooks/data_center_siting.ipynb
# Run All — full execution takes ~3-5 minutes (Monte Carlo + sensitivity sweep)
```

That's it. All input data — Cambium scenarios, AEO Table 8, and the EIA-930 / Open-Meteo parquet caches — are committed to the repo, so the notebook runs end-to-end on a fresh clone with no API key required.

### (Optional) Refresh the historical pulls

To re-pull EIA-930 demand and fuel-mix data instead of using the cached parquet files, set an EIA API key (free — register at https://www.eia.gov/opendata/register/) and pass `refresh=True`:

```bash
export EIA_API_KEY="your_key_here"
```

```python
api.fetch_all_load(refresh=True)
api.fetch_all_fuel_mix(refresh=True)
api.fetch_all_weather(refresh=True)   # Open-Meteo, no key needed
```

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` (calibrated classifier) · `matplotlib` · `pyarrow` · `requests` · public REST APIs (EIA-930, Open-Meteo)

---

## Data attribution

This project uses public data from the following sources. All data files are included in `data/raw/` under their original licenses; please preserve the attributions below in any derivative work.

- **U.S. Energy Information Administration — Annual Energy Outlook 2026, Table 8.**

  Public domain (U.S. federal government work).

  Direct download: https://www.eia.gov/outlooks/aeo/data/browser/#/?id=8-AEO2026

- **U.S. Energy Information Administration — EIA-930 Hourly Grid Monitor.**

  Public domain. Accessed via the EIA Open Data API (https://www.eia.gov/opendata/).

- **National Renewable Energy Laboratory — Cambium 2024 Datasets.**

  Free use and copy with required attribution to DOE / NREL / ALLIANCE.

  Direct download: https://scenarioviewer.nlr.gov/?project=5c7bef16-7e38-4094-92ce-8b03dfa93380&mode=download&layout=Default

  Suggested citation: Gagnon, Pieter; Pedro Andres Sanchez Perez; Julian Florez; James Morris; Marck Llerena Velasquez; and Jordan Eisenman. *Cambium 2024 Data*. National Renewable Energy Laboratory. https://scenarioviewer.nrel.gov

- **Open-Meteo — Historical Weather API.**

  Licensed under CC-BY 4.0. https://open-meteo.com/en/docs/historical-weather-api

- **Lawrence Berkeley National Laboratory — Rand et al., *Queued Up: 2025 Edition* (December 2025).**

  Public, DOE-funded. https://emp.lbl.gov/queues

---

## Acknowledgments

Course staff: Prof. Todd Strauss (INDENG 290 LEC 005, UC Berkeley) for substantive feedback on the project proposal. Teammates: Ruixin Wang, Jasmine Chen, David Zhao.

---

## License

Code in this repository is released under the **MIT License** ([LICENSE](LICENSE)). Third-party data files retain their original licenses as noted above.
