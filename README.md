# 📈 Stock Price Prediction — Time Series Forecasting with ARIMA, SARIMA, Prophet & LSTM

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%20%7C%20SARIMA-9B59B6?style=flat-square)
![Prophet](https://img.shields.io/badge/Meta-Prophet-0866FF?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Metrics-green?style=flat-square&logo=scikit-learn)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=flat-square&logo=tableau)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

> A **multi-model time series forecasting pipeline** built to predict the closing stock price of **Tata Global Beverages (NSE: TATAGLOBAL)** — benchmarking four approaches: **ARIMA**, **SARIMA**, **Facebook Prophet**, and a **stacked LSTM neural network**. The pipeline covers end-to-end data loading, preprocessing, train/test splitting, model training, forecasting, and RMSE/MAE evaluation across all four models, with a companion **Tableau workbook** (6 sheets) for exploratory visual analysis of the raw OHLC stock data.

---

## 📌 Problem Statement

Stock price prediction is a classic time series challenge where traditional statistical models, modern decomposition-based approaches, and deep learning architectures each offer different trade-offs between interpretability, seasonality handling, and long-range pattern capture. This project builds a benchmark pipeline that:

- **Forecasts the future closing price** of Tata Global Beverages from NSE historical OHLC data spanning 8 years
- **Compares four distinct forecasting models** — ARIMA (autoregressive), SARIMA (seasonal), Prophet (trend + seasonality decomposition), and stacked LSTM (sequence-learning neural network)
- **Quantifies model performance** using RMSE and MAE on a held-out 10% test set
- **Visualises raw stock behaviour** across six analytical sheets in a Tableau workbook

---

## 🎯 Objectives

- Load and preprocess NSE historical OHLC data for TATAGLOBAL (2,035 trading days, July 2010 – September 2018)
- Parse and sort by `Date`, set it as a datetime index, and isolate the `Close` price series
- Split data into a **90% train / 10% test** split for consistent model evaluation
- Train and forecast with **ARIMA(5,1,0)** — autoregressive model with one degree of differencing
- Train and forecast with **SARIMA(2,1,2)(1,1,1,5)** — seasonal ARIMA with a weekly (s=5) seasonal component
- Train and forecast with **Meta Prophet** — decomposition model with daily seasonality enabled
- Train and forecast with a **stacked LSTM** — two LSTM layers (50 units each) with a Dense output layer, trained on 60-day look-back windows
- Evaluate all four models on the test set using **RMSE and MAE**
- Visualise each model's forecast against actual Close prices
- Explore raw OHLC, volume, and turnover data interactively via a **Tableau workbook (6 sheets)**

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `NSE-TATAGLOBAL.csv` |
| Stock | Tata Global Beverages — NSE ticker: TATAGLOBAL |
| Rows | 2,035 trading days |
| Date Range | 21 July 2010 — 28 September 2018 |
| Target Variable | `Close` — daily closing price (INR) |

### Dataset Columns

| Column | Description |
|---|---|
| `Date` | Trading date — parsed to datetime and set as DataFrame index |
| `Open` | Opening price of the trading session (INR) |
| `High` | Intraday high price (INR) |
| `Low` | Intraday low price (INR) |
| `Last` | Last traded price of the session (INR) |
| `Close` | **Target** — closing price of the trading session (INR) |
| `Total Trade Quantity` | Total number of shares traded in the session |
| `Turnover (Lacs)` | Total session turnover value in Indian Lacs (₹) |

### Close Price Statistics

| Metric | Value |
|---|---|
| Mean | ₹149.45 |
| Std Dev | ₹48.71 |
| Min | ₹80.95 |
| 25th Percentile | ₹120.05 |
| Median | ₹141.25 |
| 75th Percentile | ₹156.90 |
| Max | ₹325.75 |

---

## 🔬 Methodology

```
NSE-TATAGLOBAL.csv
       │
       ▼
Data Loading & Preprocessing
       │   ├── pd.read_csv → parse Date column to datetime
       │   ├── Sort by Date ascending
       │   ├── Set Date as DataFrame index
       │   └── Isolate Close price series → close_data
       │
       ▼
Train / Test Split  (90% / 10%)
       │   ├── Train: first 1,831 trading days  (Jul 2010 – ~May 2018)
       │   └── Test:  last  204  trading days   (~May 2018 – Sep 2018)
       │
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼
Model 1 — ARIMA(5, 1, 0)                  Model 2 — SARIMA(2,1,2)(1,1,1,5)
       │   ├── Fit on train Close series         │   ├── Seasonal order s=5 (weekly cycle)
       │   ├── Forecast len(test) steps          │   ├── Fit on train Close series
       │   └── Evaluate: RMSE, MAE               │   ├── Forecast len(test) steps
       │                                         │   └── Evaluate: RMSE, MAE
       │
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼
Model 3 — Facebook Prophet                 Model 4 — Stacked LSTM
       │   ├── Rename: Date→ds, Close→y          │   ├── MinMaxScaler → scale Close [0,1]
       │   ├── Prophet(daily_seasonality=True)    │   ├── Build 60-day look-back sequences
       │   ├── make_future_dataframe(periods=     │   │       X shape: (samples, 60, 1)
       │   │         len(test))                   │   ├── Architecture:
       │   ├── predict() → extract yhat           │   │       LSTM(50, return_sequences=True)
       │   └── Evaluate: RMSE, MAE                │   │       LSTM(50)
       │                                          │   │       Dense(1)
       │                                          │   ├── Compile: adam, loss=MSE
       │                                          │   ├── Fit: epochs=10, batch_size=32
       │                                          │   ├── Predict → inverse_transform
       │                                          │   └── Evaluate: RMSE, MAE
       │
       ▼
Model Comparison Summary
       └── Print RMSE & MAE for all four models side by side
```

---

## 📊 Results

| Model | Configuration | Evaluation |
|---|---|---|
| ARIMA | order=(5, 1, 0) | RMSE & MAE on 204-day test set |
| SARIMA | order=(2,1,2), seasonal_order=(1,1,1,5) | RMSE & MAE on 204-day test set |
| Prophet | daily_seasonality=True | RMSE & MAE on 204-day test set |
| LSTM | 2×LSTM(50) + Dense(1), 60-day window, 10 epochs | RMSE & MAE on 204-day test set |

> 📝 *Refer to `Time_Series_Forecasting.ipynb` for full numeric RMSE and MAE values, all four forecast vs. actual plots, the Prophet decomposition chart, and the LSTM model summary.*

---

## 📉 Visualisations

### Notebook Plots

| Plot | Description |
|---|---|
| Close Price Over Time | Full 8-year historical Close price series with date axis and grid |
| ARIMA Forecast | Actual vs. ARIMA(5,1,0) predicted Close on the 204-day test window |
| SARIMA Forecast | Actual vs. SARIMA(2,1,2)(1,1,1,5) predicted Close on the test window |
| Prophet Forecast | Full Prophet trend + uncertainty intervals decomposition chart |
| LSTM Forecast | Actual vs. stacked LSTM predicted Close on the test window |

### Tableau Workbook

The companion workbook (`Time_Series_Forecasting.twb`) connects directly to `NSE-TATAGLOBAL.csv` and provides **6 analytical sheets** for interactive EDA of the raw OHLC data — exploring price trends, volume patterns, intraday spread, and turnover behaviour across the full 8-year history.

---

## 💡 Key Insights

- **ARIMA(5,1,0)** captures autoregressive momentum through differencing (I=1) and is fast to train, but has no seasonality component — making it well-suited for short-horizon forecasts on trending, near-stationary data
- **SARIMA(2,1,2)(1,1,1,5)** extends ARIMA with a weekly seasonal component (s=5 trading days), better handling periodic patterns in the stock's intraday and weekly price cycles that ARIMA cannot model
- **Prophet** decomposes the series into trend, seasonality, and holiday components — producing human-interpretable uncertainty intervals and automatically handling missing trading days, making it robust to gaps in financial data
- **Stacked LSTM** is the only model in the pipeline capable of learning complex, non-linear temporal dependencies across 60-day rolling windows — at the cost of higher compute and a need for careful sequence construction and inverse-scaling
- **MinMaxScaling to [0,1]** before LSTM training is essential — raw INR close prices in the ₹80–₹325 range cause gradient instability without normalisation, and inverse-transforming predictions restores interpretable price values
- **90/10 train-test split** on 2,035 days gives ~204 test samples — enough for stable metric estimation while maximising the training history available to all four models
- **Tableau EDA** surface patterns that inform modelling choices — volume spikes, intraday spread widening, and turnover anomalies all signal potential structural breaks in the price series before any model is trained

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas | Data loading, date parsing, datetime indexing, train/test splitting |
| NumPy | Array operations, 60-day sequence construction, RMSE computation |
| Matplotlib / Seaborn | Historical price plots, forecast vs. actual visualisations |
| Statsmodels | ARIMA and SARIMAX model fitting and multi-step forecasting |
| Meta Prophet | Trend + seasonality decomposition forecasting with uncertainty intervals |
| TensorFlow / Keras | Sequential stacked LSTM — layers, compile, fit, predict |
| Scikit-learn | MinMaxScaler, mean_squared_error, mean_absolute_error |
| Tableau Desktop 2025.2 | 6-sheet interactive OHLC exploratory analysis workbook |
| Jupyter Notebook | Interactive development, model training, and comparison |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet tensorflow scikit-learn jupyter
```

> ⚠️ **Prophet installation note:** Prophet requires `pystan` as a backend. If `pip install prophet` fails, try:
> ```bash
> conda install -c conda-forge prophet
> ```
> or:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Stock-Price-Prediction-Using-Time-Series-Forecasting.git

# Navigate to the project directory
cd Stock-Price-Prediction-Using-Time-Series-Forecasting

# Launch Jupyter Notebook
jupyter notebook Time_Series_Forecasting.ipynb
```

> ⚠️ Ensure `NSE-TATAGLOBAL.csv` is in the same directory as the notebook before running — it is loaded with a relative path (`pd.read_csv("NSE-TATAGLOBAL.csv")`).

### Open the Tableau Workbook

Open `Time_Series_Forecasting.twb` in **Tableau Desktop 2025.2** or later. If prompted, update the data source path to point to `NSE-TATAGLOBAL.csv` in your local project directory.

---

## 📁 Project Structure

```
📦 Stock-Price-Prediction-Using-Time-Series-Forecasting
 ┣ 📓 Time_Series_Forecasting.ipynb   # Full pipeline — ARIMA, SARIMA, Prophet, LSTM (31 cells)
 ┣ 📄 NSE-TATAGLOBAL.csv             # NSE OHLC dataset (2,035 trading days, 2010–2018)
 ┣ 📊 Time_Series_Forecasting.twb    # Tableau workbook — 6-sheet OHLC exploratory analysis
 ┣ 📄 requirements.txt               # Python dependencies
 ┗ 📄 README.md                      # Project documentation
```

---

## ⚠️ Disclaimer

> This project is built purely for **educational and analytical purposes** as a machine learning and time series case study. The forecasting models and results are derived from historical NSE data and are intended to demonstrate ARIMA, SARIMA, Prophet, and LSTM proficiency — **not** to serve as financial advice or production stock trading tooling. Past stock performance does not guarantee future results.

---

## 🔭 Future Scope

- Perform **ADF (Augmented Dickey-Fuller) and KPSS stationarity testing** to determine the differencing order `d` empirically rather than by assumption
- Use **ACF and PACF plots** to select optimal ARIMA `p` and `q` parameters systematically instead of fixed values
- Apply **auto_arima** (via `pmdarima`) to exhaustively tune ARIMA and SARIMA hyperparameters using AIC/BIC criteria
- Extend the LSTM to a **Bidirectional LSTM** or add **Dropout and BatchNormalization layers** for improved regularisation
- Incorporate additional input features — `Open`, `High`, `Low`, `Total Trade Quantity` — into a **multivariate LSTM** for richer sequence learning beyond just Close price
- Add **directional accuracy** as a financial metric — the percentage of correctly predicted up/down days alongside RMSE and MAE
- Integrate **external signals** such as the NIFTY 50 index, macroeconomic indicators, or financial news sentiment via NLP
- Build a **Streamlit dashboard** combining all four model forecasts, confidence intervals, and Tableau-style EDA charts in a single interactive web app

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

> ⭐ If you found this project useful, consider giving it a star!
