"""
predict_and_store_pytorch.py

Updated: fixed pandas deprecation warnings and timezone handling.

Note: replace DB_USER / DB_PASSWORD with your credentials.
Uploaded design doc (local path): /mnt/data/Stock Price Prediction Pipeline with MySQL and LSTM.docx
"""

import os
import logging
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from pandas import DatetimeTZDtype  # used for dtype checks
from pandas.api import types as pdt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# -------------------------
# CONFIGURATION (edit)
# -------------------------
DB_USER = "root"
DB_PASSWORD = "HelloMello@13"   # <-- put your real password here (may contain @)
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "StockData"

# Local timezone for DB timestamps (change if DB timestamps are in another zone)
LOCAL_TZ = "Asia/Kolkata"

# Uploaded doc path (local)
UPLOADED_DOC_PATH = "/mnt/data/Stock Price Prediction Pipeline with MySQL and LSTM.docx"

# Build a URL object to avoid URL-encoding issues with special characters in password
db_url = URL.create(
    drivername="mysql+mysqlconnector",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Create engine using the URL object
engine = create_engine(db_url, echo=False, pool_pre_ping=True)

STOCKS = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
          'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'ITC']

YFINANCE_SUFFIX = ".NS"
LOOKBACK_DAYS = 30
YF_INTERVAL = "5m"

TIME_STEPS = 10     # number of past points (10 * 5min = 50min)
EPOCHS = 30
BATCH_SIZE = 64
MODEL_DIR = "models"
RETRAIN = True      # if False, will load existing model if present

MIN_SAMPLES = TIME_STEPS + 5   # 15 by default

# How many seconds ahead we predict (5 minutes)
PRED_SECONDS = 5 * 60

os.makedirs(MODEL_DIR, exist_ok=True)

# Suppress a small number of benign warnings from libraries (optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
# We will not suppress DeprecationWarning from pandas globally; instead fix code below.

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# -------------------------
# PyTorch model
# -------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.15, batch_first=True):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, seq_len, features] (batch_first=True)
        out, _ = self.lstm(x)                # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]                  # take last time step -> [batch, hidden_size]
        out = self.dropout(out)
        out = self.fc(out)                   # [batch, 1]
        return out

# -------------------------
# Utilities & robust yfinance fetch (timezone-correct)
# -------------------------
def create_tables_if_not_exists(engine):
    create_stockprices = """
    CREATE TABLE IF NOT EXISTS StockPrices (
        id INT AUTO_INCREMENT PRIMARY KEY,
        stock_name VARCHAR(64) NOT NULL,
        price DECIMAL(12,4) NOT NULL,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_stock_time (stock_name, timestamp)
    );
    """
    create_predictions = """
    CREATE TABLE IF NOT EXISTS Predictions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        stock_name VARCHAR(64) NOT NULL,
        predicted_price DECIMAL(12,4) NOT NULL,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_pred_stock_time (stock_name, timestamp)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_stockprices))
        conn.execute(text(create_predictions))
    logging.info("Ensured tables exist.")

def normalize_db_timestamps(series):
    """
    Normalize DB timestamp Series (assumed tz-naive in LOCAL_TZ) to UTC-naive datetimes.
    Steps:
      - parse to_datetime
      - if tz-naive: localize to LOCAL_TZ -> convert to UTC -> tz_localize(None)
      - if tz-aware: convert to UTC -> tz_localize(None)
    """
    s = pd.to_datetime(series, errors="coerce")
    # If tz-naive, localize to LOCAL_TZ; else convert to UTC
    try:
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # tz-naive
            s = s.dt.tz_localize(LOCAL_TZ, ambiguous='infer', nonexistent='shift_forward').dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # fallback: best-effort conversion to naive datetime
        s = pd.to_datetime(series, errors="coerce")
    return s

def normalize_yf_timestamps(series):
    """
    Normalize yfinance timestamp Series to UTC-naive datetimes.
    Steps:
      - parse to_datetime
      - if tz-aware: convert to UTC -> tz_localize(None)
      - if tz-naive: assume UTC and drop tz
    """
    s = pd.to_datetime(series, errors="coerce")
    try:
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # assume tz-naive is UTC
            s = s.dt.tz_localize("UTC").dt.tz_localize(None)
    except Exception:
        s = pd.to_datetime(series, errors="coerce")
    return s

def fetch_scraped_from_db(engine, stock):
    """
    Fetch scraped rows from DB and normalize their timestamps to UTC-naive.
    """
    sql = text("SELECT timestamp, price FROM StockPrices WHERE stock_name = :s ORDER BY timestamp ASC")
    df = pd.read_sql(sql, engine, params={"s": stock})
    if df is None or df.empty:
        return pd.DataFrame(columns=['timestamp','price'])
    # Normalize DB timestamps (assume DB DATETIME are LOCAL_TZ naive)
    # Ensure columns are named 'timestamp' and 'price'
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]:'timestamp', df.columns[1]:'price'})
    df['timestamp'] = normalize_db_timestamps(df['timestamp'])
    df = df.dropna(subset=['timestamp','price']).reset_index(drop=True)
    return df[['timestamp','price']]

def _flatten_multiindex_columns(df):
    """If df has MultiIndex columns (observed with some yfinance outputs), flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['__'.join([str(x) for x in col]).strip() for col in df.columns.values]
    return df

def _resample_1m_to_5m(df_1m):
    """
    Given a df with a DatetimeIndex or a datetime column and numeric price column,
    resample to 5-minute bars (last price in window). Return ['timestamp','price'] with UTC-naive times.
    """
    df = df_1m.copy()
    df = _flatten_multiindex_columns(df)

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        for ts_col in ['Datetime','DatetimeIndex','datetime','Date','timestamp','index']:
            if ts_col in df.columns:
                df.index = pd.to_datetime(df[ts_col], errors='coerce')
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        # try first non-numeric column
        non_numeric = [c for c in df.columns if not pdt.is_numeric_dtype(df[c])]
        if non_numeric:
            df.index = pd.to_datetime(df[non_numeric[0]], errors='coerce')

    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    # Convert index to UTC-aware then resample
    idx = df.index
    try:
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
    except Exception:
        # fallback: attempt to localize to UTC
        try:
            idx = idx.tz_localize("UTC")
        except Exception:
            pass
    df.index = idx

    # find numeric price column (exclude Volume)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != 'volume']
    if not numeric_cols:
        return pd.DataFrame()
    price_col = numeric_cols[0]

    # Resample to 5 minutes using last available price in each window
    s = df[price_col].resample('5min').last().dropna()
    if s.empty:
        return pd.DataFrame()
    out = s.reset_index()
    # Ensure columns are ['timestamp','price']
    out.columns = ['timestamp', 'price']
    # Normalize timestamp to UTC-naive
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return out[['timestamp','price']]

def fetch_yfinance_history(stock):
    """
    Robustly fetch 5-minute history for `stock` via yfinance and return DataFrame with ['timestamp','price']
    with UTC-naive timestamps.
    Strategy:
      1) Try yf.download with interval='5m' for various periods
      2) If insufficient, try 1m downloads and resample to 5m
      3) Try yf.Ticker(...).history as fallback
    """
    ticker = stock + YFINANCE_SUFFIX

    def standardize_df(raw_df):
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()
        df = raw_df.copy()
        df = _flatten_multiindex_columns(df)
        # If index is DatetimeIndex and contains a numeric price column
        if isinstance(df.index, pd.DatetimeIndex):
            # make index UTC-aware (treat naive as UTC)
            idx = df.index
            try:
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
            except Exception:
                try:
                    idx = idx.tz_localize("UTC")
                except Exception:
                    pass
            df.index = idx
            # find price column
            price_cols = [c for c in df.columns if c.lower() in ('close','adj close','adj_close','adjclose')]
            if price_cols:
                price_col = price_cols[0]
            else:
                numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric = [c for c in numeric if c.lower() != 'volume']
                if not numeric:
                    return pd.DataFrame()
                price_col = numeric[0]
            out = df[[price_col]].reset_index()
            out.columns = ['timestamp', 'price']
            out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            out = out.dropna(subset=['price']).reset_index(drop=True)
            return out[['timestamp','price']]

        # If timestamp present as column
        ts_candidates = [c for c in df.columns if pdt.is_datetime64_any_dtype(df[c]) or isinstance(df[c].dtype, DatetimeTZDtype)]
        ts_col = ts_candidates[0] if ts_candidates else None
        if ts_col is None:
            for alt in ['Datetime','Date','timestamp','index']:
                if alt in df.columns:
                    ts_col = alt
                    break
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() != 'volume']
        if ts_col and numeric_cols:
            out = df[[ts_col, numeric_cols[0]]].rename(columns={ts_col:'timestamp', numeric_cols[0]:'price'}).copy()
            out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
            # normalize timestamp: tz-aware -> convert to UTC then drop tz; tz-naive -> assume UTC
            try:
                if isinstance(out['timestamp'].dtype, DatetimeTZDtype):
                    out['timestamp'] = out['timestamp'].dt.tz_convert("UTC").dt.tz_localize(None)
                else:
                    out['timestamp'] = out['timestamp'].dt.tz_localize("UTC").dt.tz_localize(None)
            except Exception:
                out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
            out = out.dropna(subset=['timestamp','price']).reset_index(drop=True)
            return out[['timestamp','price']]

        return pd.DataFrame()

    # Attempt direct 5m downloads
    try_periods = [f"{LOOKBACK_DAYS}d", "60d", "30d", "14d", "7d", "5d"]
    for p in try_periods:
        try:
            logging.info(f"yfinance: trying 5-min download for {ticker} period={p}")
            raw = yf.download(ticker, period=p, interval=YF_INTERVAL, progress=False, threads=False, auto_adjust=True)
            df5 = standardize_df(raw)
            if not df5.empty:
                logging.info(f"yfinance: 5-min attempt period={p} returned {len(df5)} rows for {ticker}")
                return df5
        except Exception as e:
            logging.warning(f"yfinance 5-min download failed for {ticker} period={p}: {e}")

    # Attempt 1m downloads and resample to 5m
    one_min_periods = ["7d", "5d", "2d", "1d"]
    for p in one_min_periods:
        try:
            logging.info(f"yfinance: trying 1-min download for {ticker} period={p}")
            raw_1m = yf.download(ticker, period=p, interval="1m", progress=False, threads=False, auto_adjust=True)
            if raw_1m is None or raw_1m.empty:
                logging.info(f"yfinance: 1-min returned no rows for {ticker} period={p}")
                continue
            df5_from_1m = _resample_1m_to_5m(raw_1m)
            if not df5_from_1m.empty:
                logging.info(f"yfinance: resampled from 1-min (period={p}) -> {len(df5_from_1m)} rows for {ticker}")
                return df5_from_1m
        except Exception as e:
            logging.warning(f"yfinance 1-min download/resample failed for {ticker} period={p}: {e}")

    # Try ticker.history fallback
    try:
        logging.info(f"yfinance: trying Ticker.history for {ticker}")
        t = yf.Ticker(ticker)
        raw = t.history(period=f"{LOOKBACK_DAYS}d", interval=YF_INTERVAL, auto_adjust=True)
        df5 = standardize_df(raw)
        if not df5.empty:
            logging.info(f"yfinance: ticker.history returned {len(df5)} rows for {ticker}")
            return df5
    except Exception as e:
        logging.warning(f"yfinance ticker.history failed for {ticker}: {e}")

    # Last resort: try to fetch a current market price
    try:
        t = yf.Ticker(ticker)
        info_price = None
        try:
            info_price = t.info.get('regularMarketPrice', None)
        except Exception:
            # some yfinance versions throw on .info; ignore
            pass
        if info_price is not None:
            now = pd.Timestamp.utcnow()
            single = pd.DataFrame({'timestamp':[now], 'price':[info_price]})
            single['timestamp'] = pd.to_datetime(single['timestamp'], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
            logging.info(f"yfinance: used info.regularMarketPrice for {ticker} = {info_price}")
            return single
    except Exception:
        pass

    logging.warning(f"yfinance: no intraday data found for {ticker}. Returning empty DataFrame.")
    return pd.DataFrame()

def combine_data(df_yf, df_db):
    """
    Both df_yf and df_db must already have 'timestamp' column of UTC-naive datetime.
    Concatenate, drop duplicates, and sort.
    """
    if df_yf is None:
        df_yf = pd.DataFrame(columns=['timestamp','price'])
    if df_db is None:
        df_db = pd.DataFrame(columns=['timestamp','price'])

    # Defensive renaming if needed
    for df in (df_yf, df_db):
        if df is None:
            df = pd.DataFrame(columns=['timestamp','price'])
        if 'timestamp' not in df.columns and df.shape[1] >= 1:
            df.rename(columns={df.columns[0]:'timestamp'}, inplace=True)
        if 'price' not in df.columns and df.shape[1] >= 2:
            df.rename(columns={df.columns[1]:'price'}, inplace=True)

    # Convert timestamps to datetime if not already (they should be normalized by fetch functions)
    try:
        df_yf['timestamp'] = pd.to_datetime(df_yf['timestamp'], errors='coerce')
    except Exception:
        df_yf['timestamp'] = normalize_yf_timestamps(df_yf.get('timestamp', pd.Series(dtype='datetime64[ns]')))

    try:
        df_db['timestamp'] = pd.to_datetime(df_db['timestamp'], errors='coerce')
    except Exception:
        df_db['timestamp'] = normalize_db_timestamps(df_db.get('timestamp', pd.Series(dtype='datetime64[ns]')))

    df = pd.concat([df_yf, df_db], ignore_index=True)
    # Ensure timezone consistency: if tz-aware convert to UTC then drop tz; else assume UTC-naive
    try:
        if isinstance(df['timestamp'].dtype, DatetimeTZDtype):
            df['timestamp'] = df['timestamp'].dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # attempt to localize naive as UTC then drop tz
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last').reset_index(drop=True)
    return df[['timestamp','price']]

def create_sequences(scaled_values, time_steps):
    X, y = [], []
    for i in range(len(scaled_values) - time_steps):
        X.append(scaled_values[i:i + time_steps])
        y.append(scaled_values[i + time_steps])
    X = np.array(X)   # shape [samples, time_steps, features]
    y = np.array(y)   # shape [samples, features]
    return X, y

def linear_extrapolate_next(df):
    """
    Fallback predictor: linear time-based extrapolation using last two points.
    df must have 'timestamp' (datetime) and 'price' (float), sorted ascending.
    Returns predicted price (float).
    """
    if df is None or df.empty:
        return None
    n = len(df)
    if n == 1:
        last_price = float(df['price'].iloc[-1])
        logging.info("Fallback used: only 1 point available — returning last price.")
        return last_price
    # use last two rows
    last_ts = pd.to_datetime(df['timestamp'].iloc[-1])
    prev_ts = pd.to_datetime(df['timestamp'].iloc[-2])
    last_price = float(df['price'].iloc[-1])
    prev_price = float(df['price'].iloc[-2])
    delta_seconds = (last_ts - prev_ts).total_seconds()
    if delta_seconds == 0:
        logging.info("Fallback used: last two timestamps are identical — returning last price.")
        return last_price
    slope_per_sec = (last_price - prev_price) / delta_seconds
    predicted = last_price + slope_per_sec * PRED_SECONDS
    logging.info(f"Fallback linear extrapolation: last={last_price}, prev={prev_price}, slope_sec={slope_per_sec:.8f}, predicted={predicted:.4f}")
    return float(predicted)

def train_model_pytorch(model, train_loader, val_loader, epochs, lr=1e-3, patience=6, model_path=None):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad()
            out = model(xb).squeeze(1)     # [batch]
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                out = model(xb).squeeze(1)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        logging.info(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.6f} val_loss: {avg_val_loss:.6f}")

        # Early stopping / checkpointing
        if avg_val_loss < best_val_loss - 1e-8:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            if model_path:
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved best model to {model_path} (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} val_loss {best_val_loss:.6f}")
                break

    # Load best model if saved
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        except Exception as e:
            logging.warning(f"Failed to load saved best model: {e}")
    return model

# -------------------------
# Main pipeline
# -------------------------
def main():
    logging.info("Starting PyTorch-based prediction pipeline.")

    create_tables_if_not_exists(engine)

    # Truncate Predictions table before inserting
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE Predictions"))
    logging.info("Truncated Predictions table.")

    results = []

    for stock in STOCKS:
        logging.info(f"Processing {stock} ...")

        df_db = fetch_scraped_from_db(engine, stock)
        df_yf = fetch_yfinance_history(stock)

        # If both empty, skip
        if (df_db is None or df_db.empty) and (df_yf is None or df_yf.empty):
            logging.warning(f"No data for {stock} (both yfinance and DB empty). Skipping.")
            continue

        # Combine data; both sources should have UTC-naive timestamps after normalization
        df_all = combine_data(df_yf, df_db)
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)

        if df_all.shape[0] < MIN_SAMPLES:
            logging.warning(f"Not enough samples for {stock}: have {df_all.shape[0]} need >= {MIN_SAMPLES}. Using fallback predictor.")
            pred_price = linear_extrapolate_next(df_all)
            if pred_price is None:
                logging.warning(f"Fallback failed for {stock}. Skipping.")
                continue
            pred_price_rounded = float(round(pred_price, 4))
            logging.info(f"{stock} fallback predicted next 5-min price: {pred_price_rounded}")
            results.append({"stock_name": stock, "predicted_price": pred_price_rounded})
            continue

        # Enough data -> train/predict with LSTM
        values = df_all['price'].astype(float).values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        X, y = create_sequences(scaled, TIME_STEPS)
        # y currently shape (samples, 1), flatten to (samples,)
        y = y.reshape(-1, )

        # Train/val split (no shuffle)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, shuffle=False)

        # Convert to torch tensors
        X_train_t = torch.tensor(X_train).float()
        y_train_t = torch.tensor(y_train).float()
        X_val_t = torch.tensor(X_val).float()
        y_val_t = torch.tensor(y_val).float()

        # DataLoaders
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model_path = os.path.join(MODEL_DIR, f"{stock}_lstm.pt")
        model = StockLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.15, batch_first=True)

        if (not RETRAIN) and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                logging.info(f"Loaded model for {stock} from {model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model for {stock}: {e}. Will train.")

        # If model not loaded or RETRAIN True -> train
        need_training = RETRAIN or (not os.path.exists(model_path))
        if need_training:
            logging.info(f"Training PyTorch model for {stock} (train_samples={len(X_train)}) ...")
            model = train_model_pytorch(model, train_loader, val_loader, epochs=EPOCHS,
                                       lr=1e-3, patience=6, model_path=model_path)
            logging.info(f"Training finished for {stock}.")

        # Prepare last sequence for prediction (latest TIME_STEPS)
        last_seq = scaled[-TIME_STEPS:]   # shape [TIME_STEPS, 1]
        last_seq = last_seq.reshape((1, TIME_STEPS, 1))  # [1, seq_len, features]
        last_seq_t = torch.tensor(last_seq).float().to(device)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_seq_t).cpu().numpy().reshape(-1, 1)  # shape [1, 1]

        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        pred_price_rounded = float(round(float(pred_price), 4))
        logging.info(f"{stock} predicted next 5-min price: {pred_price_rounded}")

        results.append({
            "stock_name": stock,
            "predicted_price": pred_price_rounded
        })

    # Insert predictions
    if results:
        try:
            with engine.begin() as conn:
                insert_sql = text("INSERT INTO Predictions (stock_name, predicted_price) VALUES (:s, :p)")
                for row in results:
                    conn.execute(insert_sql, {"s": row["stock_name"], "p": row["predicted_price"]})
            logging.info(f"Inserted {len(results)} predictions into Predictions table.")
        except Exception as e:
            logging.error(f"Failed to insert predictions: {e}")
    else:
        logging.info("No predictions generated.")

    logging.info("PyTorch pipeline finished.")

if __name__ == "__main__":
    main()
