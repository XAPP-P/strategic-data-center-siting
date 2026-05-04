"""
Project configuration: paths, market mappings, modeling constants.

Centralizing these here so the notebook stays clean and any future
changes (e.g. swapping a Cambium GEA or adjusting the planning horizon)
happen in one place.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Expected raw filenames (drop your downloads here with these names)
AEO_TABLE_8 = DATA_RAW / "Table_8._Electricity_Supply_Disposition_Prices_and_Emissions.csv"

CAMBIUM_MIDCASE_MH = DATA_RAW / "Cambium24_MidCase_month-hour_balancingArea.csv"
CAMBIUM_HIGHDEMAND_MH = DATA_RAW / "Cambium24_HighDemandGrowth_month-hour_balancingArea.csv"
CAMBIUM_HIGHNG_MH = DATA_RAW / "Cambium24_HighNGPrice_month-hour_balancingArea.csv"
CAMBIUM_LOWNG_MH = DATA_RAW / "Cambium24_LowNGPrice_month-hour_balancingArea.csv"
CAMBIUM_ANNUAL_ALL = DATA_RAW / "Cambium24_allScenarios_annual_balancingArea.csv"

EIA930_DIR = DATA_RAW / "eia930"   # one parquet per BA, written by api_clients
WEATHER_DIR = DATA_RAW / "weather"  # one parquet per market

# -----------------------------------------------------------------------------
# Five candidate markets and their mappings
# -----------------------------------------------------------------------------
MARKETS = ["Ashburn_VA", "Dallas_TX", "Phoenix_AZ", "Atlanta_GA", "Chicago_IL"]

# Display names for tables and plots
MARKET_DISPLAY = {
    "Ashburn_VA": "Ashburn, VA",
    "Dallas_TX": "Dallas, TX",
    "Phoenix_AZ": "Phoenix, AZ",
    "Atlanta_GA": "Atlanta, GA",
    "Chicago_IL": "Chicago, IL",
}

# Cambium GEA region (the 18-region aggregation used in the
# month-hour and annual files we have)
MARKET_TO_GEA = {
    "Ashburn_VA": "PJM_East",          # Dominion Energy zone in PJM
    "Dallas_TX": "ERCOT",              # Texas (most of state)
    "Phoenix_AZ": "WestConnect_South", # AZ + NM in WECC
    "Atlanta_GA": "SERTP",             # Southern Co / Georgia Power
    "Chicago_IL": "PJM_West",          # ComEd zone in PJM
}

# EIA-930 Balancing Authority code (for hourly load + fuel mix).
# Note: PJM is reported as a single BA in EIA-930 with subregion data
# available via the subregion endpoint; for Dominion (DOM) and ComEd
# (COMD) we hit /electricity/rto/region-sub-ba-data/.
EIA930_BA = {
    "Ashburn_VA": ("PJM", "DOM"),     # (parent BA, subregion)
    "Dallas_TX": ("ERCO", None),
    "Phoenix_AZ": ("AZPS", None),
    "Atlanta_GA": ("SOCO", None),
    "Chicago_IL": ("PJM", "CE"),
}

# Geographic coordinates for Open-Meteo weather pulls (city center / major
# data center cluster). Time zones used for hourly index alignment.
MARKET_LOCATION = {
    "Ashburn_VA":  {"lat": 39.0438, "lon":  -77.4874, "tz": "America/New_York"},
    "Dallas_TX":   {"lat": 32.7767, "lon":  -96.7970, "tz": "America/Chicago"},
    "Phoenix_AZ":  {"lat": 33.4484, "lon": -112.0740, "tz": "America/Phoenix"},
    "Atlanta_GA":  {"lat": 33.7490, "lon":  -84.3880, "tz": "America/New_York"},
    "Chicago_IL":  {"lat": 41.8781, "lon":  -87.6298, "tz": "America/Chicago"},
}

# -----------------------------------------------------------------------------
# Planning horizon
# -----------------------------------------------------------------------------
ANALYSIS_START_YEAR = 2026
ANALYSIS_END_YEAR = 2035    # inclusive; gives a 10-year window
ANALYSIS_YEARS = list(range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1))

HISTORICAL_START = "2023-01-01"
HISTORICAL_END = "2025-12-31"

# -----------------------------------------------------------------------------
# Engineering / financial assumptions
# -----------------------------------------------------------------------------
IT_LOAD_MW = 100               # nameplate IT load assumption for the data center
HOURS_PER_YEAR = 8760
DISCOUNT_RATE = 0.08

# Dynamic PUE function: piecewise quadratic above reference temp, capped.
#   T <= reference:        PUE = base
#   T > reference:         PUE = base + a*(T-ref) + b*(T-ref)^2, capped at max
# Calibrated so PUE(20°C) = 1.20, PUE(35°C) = 1.40, PUE(45°C) = 1.55,
# matching published hyperscale operator data (LBNL 2024 data center
# energy report, ASHRAE class A1 envelope).
PUE_BASE = 1.20
PUE_REFERENCE_TEMP_C = 20.0
PUE_LINEAR_PER_C = 0.008      # was 0.005 — bumped up
PUE_QUAD_PER_C2 = 0.00025     # quadratic term: super-linear above 35°C
PUE_MAX = 1.60

# -----------------------------------------------------------------------------
# Risk modeling
# -----------------------------------------------------------------------------
# Normalized stress event definition (replaces the absolute $500/MWh threshold
# from the original proposal — directly responds to Todd's feedback that
# "different markets and market designs may have scarcity events of different
# magnitudes; you may want to normalize or choose a threshold percentage").
STRESS_NETLOAD_PERCENTILE = 0.95  # top 5% of net load
STRESS_TEMP_PERCENTILE = 0.90     # top 10% of temperature

# CVaR confidence level for risk-adjusted TCO
CVAR_ALPHA = 0.95

# -----------------------------------------------------------------------------
# Monte Carlo
# -----------------------------------------------------------------------------
N_SIMULATIONS = 5000
RANDOM_SEED = 42

# Carbon price scenarios ($ per ton CO2e), discrete distribution
CARBON_SCENARIOS = {
    "no_policy": (0,   0.30),
    "moderate":  (25,  0.40),
    "aggressive":(75,  0.30),
}
