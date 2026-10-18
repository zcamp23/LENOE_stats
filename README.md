# LENOE_stats

Analytics and interactive dashboard for drag racing event statistics.

## Project Structure

```text
LENOE_stats/
│
├── data/
│   ├── raw/                   # Raw .mdb database files per year (gitignored)
│   ├── interim/               # Intermediate extracted files per year
│   └── processed/             # Clean Star Schema files (.parquet / .csv / .db)
│
├── notebooks/                 # Jupyter Notebooks for EDA, statistics & ML experiments
│   ├── 01_schema_and_data_profiling.ipynb
│   ├── 02_data_cleaning_and_anomalies.ipynb
│   ├── 03_statistical_analysis_and_lane_bias.ipynb
│   └── 04_ml_et_prediction_and_win_prob.ipynb
│
├── src/                       # Modular Python codebase
│   ├── __init__.py
│   ├── extract.py             # Database connection (.mdb via pyodbc) and extraction
│   ├── transform.py           # Cleaning, unit conversion (ms -> s), category handling
│   ├── feature_engineering.py # True RT, Back-half gain, Consistency metrics
│   └── models.py              # ML model training & ET prediction
│
├── tests/                     # Automated unit and data quality tests
│   └── test_data_quality.py   # Pytest data validation checks
│
├── artifacts/                 # Saved models (.pkl), figures, and data dictionaries
│   ├── figures/
│   └── models/
│
├── app/                       # Interactive Dashboard (Streamlit + Plotly)
│   └── streamlit_app.py
│
├── docs/
│   ├── PRD.md                 # Product Requirements Document
│   └── data_dictionary.md     # Database tables and fields documentation
│
├── .gitignore                 # Git ignore rules for raw data and cache
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview and documentation
```

## Getting Started

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Data Placement
Place raw Access database (`.mdb`) files in `data/raw/` (these files are gitignored).
