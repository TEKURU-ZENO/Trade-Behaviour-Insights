# 🚀 Trader Behavior Insights  
### Understanding Trader Performance Through Market Sentiment & Volatility  

![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![LSTM](https://img.shields.io/badge/Sequence--Model-LSTM-orange.svg)
![License](https://img.shields.io/badge/License-Private-red.svg)

---

## 📌 Project Overview

This project investigates how **Bitcoin market sentiment** (Fear–Greed Index) influences **trader performance** on Hyperliquid.  
We perform:

- Full **data cleaning & normalization**
- Rich **feature engineering** (sentiment, volatility, behavior)
- **Exploratory Data Analysis**
- **Predictive modeling** using:
  - LightGBM + Optuna
  - LSTM sequence models
- Behavioral insights on trader skill, risk, conviction, and volatility–sentiment interaction

This repository follows production-level data science structure: modular, reproducible, and scalable.

---

## 📂 Folder Structure

```
trader-behavior-insights/
│
├── data/
│   ├── raw/                # Provided datasets
│   └── processed/          # Engineered dataset (parquet)
│
├── notebooks/
│   ├── 01_data_prep.ipynb  # Cleaning + feature engineering
│   ├── 02_eda.ipynb        # Exploration + visual insights
│   └── 03_modeling.ipynb   # ML models
│
├── models/                 # Saved LightGBM & LSTM models
│
├── src/
│   ├── data_utils.py
│   ├── feature_engineering.py
│   ├── volatility.py
│   ├── modeling_advanced.py
│   └── sequence_dataset.py
│
└── README.md
```

---

## 🧹 Data Preparation

### Raw Inputs
- `historical.csv` (trader execution logs)
- `fear_greed_index.csv` (sentiment values)

### Standardized Columns
To ensure consistency across models:

| Raw Column | Standardized |
|-----------|--------------|
| Execution Price | execution_price |
| Size Tokens | size |
| Closed PnL | closedpnl |
| Timestamp / Timestamp IST | time |
| Coin | symbol |
| Account | account |

### Output
Cleaned enriched dataset saved as:

```
data/processed/trades_processed.parquet
```

---

## 🧠 Feature Engineering

### ✔ Trade-Level Features
- notional  
- return_pct  
- win  
- leverage  
- weekday / weekend  
- time_of_day  

### ✔ Sentiment Features
- score  
- classification  
- score_3d  
- score_7d  
- sentiment_shift = score_3d − score_7d  
- sentiment alignment  

### ✔ Volatility Features
Using rolling execution-price volatility per symbol:

- price_ret  
- volatility  
- volatility_bucket (quartiles)  
- sentiment × volatility interaction  

### ✔ Behavioral (Trader Skill) Features
- winrate_10 / winrate_30 / winrate_100  
- avg_return_…  
- pnl_stability_…  
- conviction  
- risk_per_trade  

---

## 📊 Exploratory Data Analysis

Key questions explored:

### 1. Does sentiment affect profitability?
- Extreme Fear correlates with higher variance in returns
- Greed periods produce larger trade sizes

### 2. How do traders behave in different volatility regimes?
- High-volatility → mean-reversion behavior emerges  
- Low-volatility → trend-following behavior increases

### 3. Can trader skill be quantified?
Yes — stable accounts show:

- Consistent winrates  
- Lower PnL volatility  
- Higher conviction in favorable regimes  

---

## 🤖 Modeling

This project includes two ML pipelines:

### 🔥 LightGBM + Optuna
- Target: Predict if a trade will be profitable  
- GroupKFold (group = account) to avoid leakage  
- Bayesian hyperparameter tuning  
- Early stopping via LightGBM callbacks  

Outputs:
- `models/lightgbm_optuna.pkl`

### 🔥 LSTM Behavioral Sequence Model
Captures temporal trader patterns using:

- Sentiment
- Notional scale
- Leverage
- Time-of-day
- Volatility conditions

Outputs:
- `models/lstm_state_dict.pt`

---

## ▶️ How to Run

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run Data Preparation
```
notebooks/01_data_prep.ipynb
```

### 3. Run EDA
```
notebooks/02_eda.ipynb
```

### 4. Run Modeling
```
notebooks/03_modeling.ipynb
```

---

## 💡 Key Insights

- **Sentiment + Volatility interaction is highly predictive**  
- High-skill traders demonstrate **low behavioral drift**  
- LSTM models reveal stateful patterns across sequential trades  
- Greed periods trigger **overconfidence**, increasing notional size  
- Fear periods improve **risk-adjusted returns** for disciplined traders  

---

## 🔧 Future Improvements

- SHAP interpretability  
- More sequence-based architectures (GRU, Transformer)  
- Real-time pipeline ingestion  
- Deployment using FastAPI + Docker  
- Trader clustering (HDBSCAN, KMeans)  

---

## 📘 License
This repository is part of a hiring assignment and is not intended for redistribution.

---

## 📩 Contact
For questions or collaboration, please reach out.


