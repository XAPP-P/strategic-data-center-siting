# Input data — sources and licenses

All input data needed to reproduce the analysis is included in this folder. Nothing needs to be downloaded separately.

## Folder contents

```
data/raw/
├── Table_8._Electricity_Supply_Disposition_Prices_and_Emissions.csv
├── Cambium24_MidCase_month-hour_balancingArea.csv
├── Cambium24_HighDemandGrowth_month-hour_balancingArea.csv
├── Cambium24_HighNGPrice_month-hour_balancingArea.csv
├── Cambium24_LowNGPrice_month-hour_balancingArea.csv
├── Cambium24_allScenarios_annual_balancingArea.csv
├── eia930/                       # Hourly demand and fuel mix (parquet caches), 2023-2025
└── weather/                      # Hourly 2-meter temperature (parquet caches), 2023-2025
```

## Sources and licenses

### `Table_8._Electricity_Supply_Disposition_Prices_and_Emissions.csv`

EIA Annual Energy Outlook 2026, Table 8: Electricity Supply, Disposition, Prices, and Emissions. Public domain (U.S. federal government work).

Direct download: https://www.eia.gov/outlooks/aeo/data/browser/#/?id=8-AEO2026

### `Cambium24_*.csv`

NREL Cambium 2024 Datasets — five scenario files at month-hour and annual resolutions, balancing-area geography. Free to use and copy with required attribution to DOE / NREL / ALLIANCE.

Direct download: https://scenarioviewer.nlr.gov/?project=5c7bef16-7e38-4094-92ce-8b03dfa93380&mode=download&layout=Default

Suggested citation: Gagnon, Pieter; Pedro Andres Sanchez Perez; Julian Florez; James Morris; Marck Llerena Velasquez; and Jordan Eisenman. *Cambium 2024 Data*. National Renewable Energy Laboratory. https://scenarioviewer.nrel.gov

### `eia930/*.parquet`

Cached responses from the EIA-930 Hourly Grid Monitor REST API: hourly demand by balancing authority and hourly net generation by fuel type, 2023-01-01 through 2025-12-31. Public domain (U.S. federal government work). Accessed via https://www.eia.gov/opendata/.

Pull logic lives in `src/api_clients.py`. To refresh, set `EIA_API_KEY` and call the helpers with `refresh=True`.

### `weather/*.parquet`

Cached responses from the Open-Meteo Historical Weather API: hourly 2-meter temperature for each market's lat/lon, 2023-01-01 through 2025-12-31. Licensed under CC-BY 4.0.

API documentation: https://open-meteo.com/en/docs/historical-weather-api
