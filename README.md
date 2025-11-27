# Stock Price Prediction Pipeline

This repository contains a complete pipeline that:

1. Collects real-time intraday stock prices (scraped by Power Automate) and stores them in a MySQL database table `StockPrices`.
2. Fetches historical intraday data from Yahoo Finance (`yfinance`) for the last 30 days (5‑minute bars) and complements it with the scraped data.
3. Trains (or loads) a PyTorch LSTM model per stock to predict the **next 5‑minute** price.
4. Stores predictions in a MySQL table `Predictions` ready for visualization (Power BI).

> **Note:** The pipeline uses PyTorch (with optional CUDA/GPU). The code normalizes timestamps to **UTC‑naive** datetimes.

---

## Table schemas (MySQL)

You can create the tables automatically: the script will create them if they don't exist. For reference:

### `StockPrices`

```sql
CREATE TABLE IF NOT EXISTS StockPrices (
  id INT AUTO_INCREMENT PRIMARY KEY,
  stock_name VARCHAR(64) NOT NULL,
  price DECIMAL(12,4) NOT NULL,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_stock_time (stock_name, timestamp)
);
```

* `timestamp` is `DATETIME` (MySQL) and will be normalized by the script assuming it is stored in a local timezone (default `Asia/Kolkata`).

### `Predictions`

```sql
CREATE TABLE IF NOT EXISTS Predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  stock_name VARCHAR(64) NOT NULL,
  predicted_price DECIMAL(12,4) NOT NULL,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_pred_stock_time (stock_name, timestamp)
);
```

* `Predictions` will be truncated at the beginning of each run to avoid clumping old results with new ones.

---

## Prerequisites

* Python 3.9+ (tested on 3.10/3.11/3.13)
* PyTorch installed with CUDA if you plan to use GPU (optional but recommended).
* MySQL server accessible (local or remote)
* Power Automate flow configured to insert real-time scraped rows into `StockPrices` every 5 minutes.

---

## Configuration

1. **Database credentials** — the script uses `sqlalchemy.engine.URL.create(...)` to build a safe connection URL (this avoids errors when your password contains `@` or other special characters).

2. **Stocks list** — edit the `STOCKS` list in the script to add/remove tickers. The code appends `.NS` to each ticker for Yahoo India exchange (e.g., `RELIANCE.NS`).

3. **Time zone handling** — by default the script assumes DB `DATETIME` values are in `LOCAL_TZ` (`Asia/Kolkata`) and converts them to UTC-naive datetimes for internal processing. Change `LOCAL_TZ` if your DB uses another timezone.

---

## High level overview

1. The script ensures the two tables (`StockPrices`, `Predictions`) exist.
2. It truncates `Predictions` so the table only contains the results from the latest run.
3. For each stock in `STOCKS`:

   * Fetch scraped rows from `StockPrices`.
   * Try to fetch intraday historical data via `yfinance`:

     * preferred: `yf.download(ticker, interval='5m', period='30d')`.
     * fallback: try other periods (`60d`,`30d`,`14d`,`7d`,`5d`).
     * if 5m data is missing, try 1m downloads and resample to 5m (this often works and is robust to Yahoo's varying responses).
     * if still missing, try `yf.Ticker(ticker).history(...)`.
     * if all fail, try getting the last market price from `t.info['regularMarketPrice']` and use a fallback predictor.
   * Normalize all timestamps to UTC-naive datetimes to avoid tz-aware/naive mixing.
   * Combine yfinance and DB-scraped rows, sort, and deduplicate timestamps.
   * If samples < `MIN_SAMPLES` the script uses a fallback linear extrapolation (or last price) and saves it.
   * Otherwise train (or load) a PyTorch LSTM model per stock and predict single-step (next 5 min).
   * Store predictions into `Predictions`.

---

## Power Automate notes

* Your Power Automate flow should append rows to the `StockPrices` table with: `stock_name`, `price` (decimal), and an optional `timestamp`. If you omit a timestamp, MySQL `CURRENT_TIMESTAMP` will be used.
* Keep the scraping frequency at or above 5 minutes to match the model interval. The script expects the scraped data is every 5 minutes and will combine it with yfinance intraday data.

---

## Model details

* **Architecture:** Stacked LSTM (2 layers), dropout, final linear layer producing single output.
* **Input:** sequences of `TIME_STEPS` (default 10) of normalized prices (MinMaxScaler on full series).
* **Output:** one-step ahead prediction (the next 5-minute price)
* **Training:** train/validation split uses `train_test_split(..., shuffle=False)` to preserve time ordering. Early stopping and best-model checkpointing are implemented.

---

## Architecture overview

```
[Power Automate] ---> (HTTPS / Web interaction / Browser scrape) ---> [Power Automate Flow] 
          |                                              |
          | every 5 minutes                               | INSERT row(s)
          v                                              v
     (web pages)                                    [MySQL Server (StockData DB)]
                                                         |  (StockPrices table)
                                                         |
              (scheduled)                                v
                           [Python script: LSTM.py]  <--- reads StockPrices
                          |        ^
                          | fetches historical     |
                          | yfinance (30d @5m)     |
                          v                        |
                    [yfinance (Yahoo)]              |
                          |                        |
                          v                        |
                 Trained/loaded PyTorch LSTM        |
                          | produces predictions   |
                          v                        |
                 Writes to Predictions table ------>
                                                         |
                                                         v
                                                    [Power BI]
                                                    (connected to MySQL Predictions table)
```

### Component responsibilities

* **Power Automate (Flow)**

  * Scrapes stock prices from chosen web sources every 5 minutes.
  * Sends the scraped row(s) to your MySQL `StockPrices` table (INSERT). Each insert contains `stock_name`, `price` and optionally `timestamp`.
  * Invokes python script `LSTM.py`.

* **MySQL Server**

  * Central persistence for both raw scraped intraday rows (`StockPrices`) and the model output (`Predictions`).
  * The Python pipeline reads `StockPrices`, and writes back a small batch into `Predictions` (table truncated first on each run).

* **Python / PyTorch pipeline** (`LSTM.py`)

  * Reads scraped rows from `StockPrices`, fetches historical intraday from Yahoo via `yfinance`, normalizes timestamps to UTC-naive, stitches both sources, trains/loads an LSTM model per-stock and produces one-step-ahead (5-minute) predictions.
  * Truncates `Predictions` table at start and writes predicted rows ready for Power BI.

* **Power BI**

  * Connects directly to the MySQL `Predictions` table to render dashboards.
