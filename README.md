# 📈 Stock Price Prediction Using Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-lightblue?style=flat-square&logo=tableau)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A time series forecasting project that predicts the stock closing price of **Tata Global Beverages (NSE: TATAGLOBAL)** using classical statistical models and machine learning techniques, complemented by an interactive Tableau dashboard for trend analysis.

---

## 📌 Problem Statement

Stock price prediction is one of the most challenging and widely studied problems in quantitative finance. Prices are influenced by countless unpredictable factors, making accurate forecasting difficult but valuable. This project applies time series analysis and machine learning to historical NSE stock data to forecast future closing prices and identify underlying market trends.

---

## 🎯 Objective

- Analyse historical stock price data for **Tata Global Beverages** listed on the NSE
- Apply time series decomposition to identify trend, seasonality, and residual components
- Build and compare forecasting models including ARIMA and ML-based regressors
- Visualise price movements, forecasts, and performance metrics via a **Tableau dashboard**

---

## 📂 Dataset

| Property | Detail |
|---|---|
| File | `NSE-TATAGLOBAL.csv` |
| Source | NSE (National Stock Exchange of India) — Tata Global Beverages historical data |
| Features | Date, Open, High, Low, Close, Volume, Turnover |
| Target Variable | `Close` — Daily closing stock price |
| Frequency | Daily trading data |

---

## 🔬 Methodology

```
Historical Stock Data (NSE-TATAGLOBAL.csv)
              │
              ▼
Exploratory Data Analysis
              │   ├── Closing price trend over time
              │   ├── Rolling mean & standard deviation
              │   ├── Volume analysis
              │   └── Stationarity check (ADF Test)
              │
              ▼
Time Series Preprocessing
              │   ├── Date indexing & resampling
              │   ├── Differencing for stationarity
              │   └── Train-test split (chronological)
              │
              ▼
Model Building
              │   ├── ARIMA / SARIMA (statistical baseline)
              │   ├── Moving Average Smoothing
              │   └── ML Regressor (feature-engineered approach)
              │
              ▼
Evaluation
              │   ├── RMSE (Root Mean Squared Error)
              │   ├── MAE (Mean Absolute Error)
              │   └── Forecast vs Actual plot
              │
              ▼
Tableau Dashboard
                  ├── Historical price trend visualisation
                  ├── Volume vs price correlation
                  └── Forecast overlay on actual data
```

---

## 📊 Results

| Model | RMSE | MAE |
|---|---|---|
| Moving Average | — | — |
| ARIMA | — | — |
| ML Regressor | — | — |
| Best Model | — | — |

> 📝 *Refer to `Time_Series_Forecasting.ipynb` for full results, forecast plots, and error metrics.*

---

## 📊 Tableau Dashboard

The repository includes `Time_Series_Forecasting.twb` — an interactive Tableau workbook featuring:

- **Price Trend View**: Historical closing price movements over the full date range
- **Volume Analysis**: Trading volume plotted alongside price to spot momentum signals
- **Forecast Overlay**: Predicted vs actual closing prices for the test period
- **Moving Averages**: 30-day and 90-day moving average overlays for trend identification

> Open with **Tableau Desktop** or **Tableau Public** to explore the dashboard interactively.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Time series data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Static visualisations |
| Statsmodels | ARIMA, ADF test, time series decomposition |
| Scikit-learn | ML-based forecasting and evaluation metrics |
| Tableau | Interactive financial dashboard |
| Jupyter Notebook | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/PJ2001-IND/Stock-Price-Prediction-Using-Time-Series-Forecasting.git

# Navigate to the project directory
cd Stock-Price-Prediction-Using-Time-Series-Forecasting

# Launch Jupyter Notebook
jupyter notebook Time_Series_Forecasting.ipynb
```

### View the Dashboard

Open `Time_Series_Forecasting.twb` in **Tableau Desktop** or upload to **Tableau Public**.

---

## 📁 Project Structure

```
📦 Stock-Price-Prediction-Using-Time-Series-Forecasting
 ┣ 📓 Time_Series_Forecasting.ipynb    # Full forecasting pipeline & analysis
 ┣ 📊 Time_Series_Forecasting.twb      # Interactive Tableau dashboard
 ┣ 📄 NSE-TATAGLOBAL.csv               # Historical stock data (NSE)
 ┗ 📄 README.md                         # Project documentation
```

---

## 💡 Key Insights

- The closing price of Tata Global Beverages shows a clear long-term upward trend with periodic volatility spikes
- The ADF test confirms the raw series is non-stationary; first-order differencing achieves stationarity
- ARIMA performs well as a short-horizon forecaster but struggles with sudden price jumps
- Volume spikes often precede significant price movements — a useful signal for feature engineering

---

## 🔭 Future Scope

- Integrate **LSTM (Long Short-Term Memory)** deep learning for improved sequence modelling
- Incorporate **sentiment analysis** from financial news as an external feature
- Build a **live dashboard** pulling real-time NSE data via the Yahoo Finance API
- Deploy as a **Streamlit web app** for interactive stock price forecasting

---

## ⚠️ Disclaimer

> This project is built purely for **educational and research purposes**. The predictions made by these models are **not financial advice** and should not be used to make real investment decisions.

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project useful, consider giving it a star!
