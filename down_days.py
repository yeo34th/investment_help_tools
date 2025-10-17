#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Rolling ATH drawdown detector (CSV-out support)
# - 초기 ATH: API 또는 로컬 계산
# - 이후 High로 롤링 ATH 갱신
# - 기본 종가(Close) 기준, --use-low 로 저가(Low)
# - 항상 원시(raw) 가격 사용
# - 기간 d/m/y/ytd/max 자유 형식
# - optional --csv-out <file>

import argparse
import sys
import json
import re
import urllib.request
from typing import Tuple
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Requires: pip install yfinance pandas", file=sys.stderr)
    sys.exit(1)

# -------------------- #
# Argument Parsing
# -------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Rolling ATH drawdown days (API-seeded init ATH optional).")
    p.add_argument("--tickers", "-t", required=True, help="Space-separated tickers, e.g. 'QQQ NVDA'")
    p.add_argument("--period", "-p", default="1y", help="Examples: 10d, 3m, 2y, ytd, max (default: 1y)")
    p.add_argument("--threshold", "-th", type=float, default=15.0,
                   help="Drawdown threshold in percent (default: 15)")
    p.add_argument("--use-low", action="store_true",
                   help="Use Low instead of Close for drawdown calculation")
    p.add_argument("--init-ath-mode", choices=["api", "local"], default="local",
                   help="Seed initial ATH from 'api' or 'local' (default: local)")
    p.add_argument("--ath-api-url", default=None,
                   help="API URL template with {ticker} and {date} (YYYY-MM-DD; date = first_day-1)")
    p.add_argument("--csv-out", default=None,
                   help="Optional: path to output CSV file (e.g. results.csv)")
    return p.parse_args()

# -------------------- #
# Utility
# -------------------- #
def normalize_period(period: str) -> str:
    s = period.strip().lower()
    if s in ("ytd", "max"):
        return s
    m = re.fullmatch(r"(\d+)\s*([dmy])", s)
    if not m:
        raise ValueError("Period must be like '10d', '3m', '2y', or 'ytd'/'max'.")
    n, unit = m.groups()
    if unit == "m":
        unit = "mo"
    return f"{n}{unit}"

def _tz_naive_index(idx) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx)
    try:
        return dt.tz_localize(None)
    except (TypeError, AttributeError):
        return dt

def _series_from_any(df: pd.DataFrame, field: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        if field in cols.get_level_values(0):
            sub = df[field]
            if isinstance(sub, pd.Series):
                return pd.to_numeric(sub, errors="coerce")
            if isinstance(sub, pd.DataFrame):
                for c in sub.columns:
                    s = pd.to_numeric(sub[c], errors="coerce")
                    if s.notna().any():
                        return s
                return pd.to_numeric(sub.iloc[:, 0], errors="coerce")
        for c in cols:
            if (isinstance(c, tuple) and field in c) or (c == field):
                s = df[c]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return pd.to_numeric(s, errors="coerce")
        if field == "Close":
            return _series_from_any(df, "Adj Close")
        raise RuntimeError(f"Missing field '{field}' (MultiIndex).")
    else:
        if field in df.columns:
            return pd.to_numeric(df[field], errors="coerce")
        if field == "Close" and "Adj Close" in df.columns:
            return pd.to_numeric(df["Adj Close"], errors="coerce")
        raise RuntimeError(f"Downloaded data lacks '{field}' column.")

def _download_ohlc(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=False, progress=False, group_by="column")
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker} (period={period}).")
    out = pd.DataFrame(index=_tz_naive_index(df.index))
    out.index.name = "Date"
    out["High"]  = _series_from_any(df, "High")
    out["Low"]   = _series_from_any(df, "Low")
    out["Close"] = _series_from_any(df, "Close")
    return out.sort_index()

def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ath-script/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

# -------------------- #
# ATH Seeders
# -------------------- #
def seed_initial_ath_via_api(ticker: str, first_day: pd.Timestamp, api_tpl: str) -> Tuple[float, pd.Timestamp]:
    if not api_tpl:
        raise RuntimeError("init-ath-mode=api requires --ath-api-url")
    cutoff = (pd.Timestamp(first_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = api_tpl.replace("{ticker}", ticker).replace("{date}", cutoff)
    data = _fetch_json(url)
    ath_val = float(data.get("ath"))
    ath_date = pd.Timestamp(str(data.get("ath_date"))).tz_localize(None)
    return ath_val, ath_date

def seed_initial_ath_locally(ticker: str, first_day: pd.Timestamp) -> Tuple[float, pd.Timestamp]:
    df_full = _download_ohlc(ticker, period="max")
    hist = df_full.loc[df_full.index < pd.Timestamp(first_day)]
    if hist.empty:
        return 0.0, pd.Timestamp(first_day).tz_localize(None)
    idx = hist["High"].idxmax()
    return float(hist.loc[idx, "High"]), pd.Timestamp(idx).tz_localize(None)

# -------------------- #
# Rolling ATH Logic
# -------------------- #
def compute_rolling_drawdowns(df: pd.DataFrame,
                              init_ath_val: float,
                              init_ath_date: pd.Timestamp,
                              threshold: float,
                              use_low: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date","Low","Close","Drawdown%","Rolling_ATH","ATH_Anchor_Date"])
    dates  = df.index.tolist()
    highs  = df["High"].astype(float).tolist()
    lows   = df["Low"].astype(float).tolist()
    closes = df["Close"].astype(float).tolist()

    rolling_ath = []
    anchor_dates = []

    first_ath = max(float(init_ath_val), float(highs[0]))
    if float(highs[0]) >= float(init_ath_val):
        current_anchor = dates[0]
    else:
        current_anchor = init_ath_date

    rolling_ath.append(first_ath)
    anchor_dates.append(current_anchor)

    for i in range(1, len(dates)):
        if highs[i] > rolling_ath[i-1]:
            rolling_ath.append(float(highs[i]))
            current_anchor = dates[i]
        else:
            rolling_ath.append(rolling_ath[i-1])
        anchor_dates.append(current_anchor)

    basis_series = pd.Series(lows if use_low else closes, index=dates, dtype="float64")
    ath_series   = pd.Series(rolling_ath, index=dates, dtype="float64")
    dd_pct = (ath_series - basis_series) / ath_series.replace(0, pd.NA) * 100.0
    dd_pct = dd_pct.astype(float).fillna(0.0)

    out = pd.DataFrame({
        "Date": dates,
        "Low":  [round(float(x), 4) for x in lows],
        "Close":[round(float(x), 4) for x in closes],
        "Drawdown%": [round(float(x), 2) for x in dd_pct],
        "Rolling_ATH": [round(float(x), 4) for x in rolling_ath],
        "ATH_Anchor_Date": [pd.Timestamp(d).strftime("%Y-%m-%d") if pd.notna(d) else "" for d in anchor_dates],
    })
    out = out.loc[out["Drawdown%"] >= threshold].reset_index(drop=True)
    return out

# -------------------- #
# Output Helpers
# -------------------- #
def to_text_table(df: pd.DataFrame, cols=None) -> str:
    if cols is None:
        cols = ["Date","Low","Close","Drawdown%","Rolling_ATH","ATH_Anchor_Date"]
    if df.empty:
        return "(no rows)"
    df_str = df.copy()
    for c in cols:
        df_str[c] = df_str[c].astype(str)
    widths = {c: max(len(c), *(len(v) for v in df_str[c])) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep    = "-+-".join("-"*widths[c] for c in cols)
    lines = []
    for i in df_str.index:
        row = " | ".join(df_str.loc[i, c].ljust(widths[c]) for c in cols)
        lines.append(row)
    return f"{header}\n{sep}\n" + "\n".join(lines)

# -------------------- #
# Main
# -------------------- #
def main():
    args = parse_args()
    try:
        period = normalize_period(args.period)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    tickers = [t.strip().upper() for t in args.tickers.split() if t.strip()]
    threshold = float(args.threshold)
    all_rows = []
    blocks = []

    for t in tickers:
        try:
            ohlc = _download_ohlc(t, period)
            if ohlc.empty:
                blocks.append(f"# {t}\n(no data)")
                continue
            first_day = ohlc.index[0]

            if args.init_ath_mode == "api":
                init_ath_val, init_ath_date = seed_initial_ath_via_api(t, first_day, args.ath_api_url)
            else:
                init_ath_val, init_ath_date = seed_initial_ath_locally(t, first_day)

            rows = compute_rolling_drawdowns(
                df=ohlc,
                init_ath_val=init_ath_val,
                init_ath_date=init_ath_date,
                threshold=threshold,
                use_low=args.use_low
            )
            rows["Ticker"] = t
            all_rows.append(rows)

            basis = "Low" if args.use_low else "Close"
            header = f"# {t} | Basis={basis} | InitATH={init_ath_val:.4f} on {pd.Timestamp(init_ath_date).date()} | Price=raw"
            blocks.append(header + "\n" + to_text_table(rows))
        except Exception as e:
            blocks.append(f"# {t}\n(error: {e})")

    print("\n\n".join(blocks))

    if args.csv_out and all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(args.csv_out, index=False)
        print(f"\n[✓] CSV saved to: {args.csv_out}")

if __name__ == "__main__":
    main()
