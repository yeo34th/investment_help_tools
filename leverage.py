#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Side-by-Side Return Table (calendar-aligned, split-safe)
- Any number of tickers side-by-side.
- Flexible periods: Nd / Nw / Nm / Ny (e.g., 10d, 3w, 3m, 2y) and 'ytd'.
- Window anchor:
    * period='ytd' -> as-of.year-01-01
    * Nd/Nw/Nm/Ny  -> as-of minus N days/weeks/months/years (calendar-accurate for m/y)
  If start is not a trading day, use the NEXT trading day's first close per ticker.
- Aggregation:
    * --interval auto    -> window >= 365 days => MONTHLY; else WEEKLY
    * --interval daily   -> DAILY (each trading day is a bucket)
    * --interval weekly  -> WEEKLY  (Monday-anchored)
    * --interval monthly -> MONTHLY (calendar months)
    * --interval yearly  -> YEARLY  (calendar years)
  Inside each bucket we compound DAILY returns (daily-compounded within bucket).
- Bucket metrics:
    Trading days = # daily rows in the bucket (shown once next to date)
    %            = ((∏(1 + daily_return)) - 1) * 100
    Cum. %       = cumulative compounded % across buckets
- Basis:
    price (default) -> Close (splits only). If a split is detected in the window,
                       auto-switch to Adjusted for the TABLE (override via --respect-price).
    total           -> Adjusted / Adj Close (dividends + splits)
- Diagnostics table:
    Per-ticker: Start, End, simple Close% vs Adj% and split note, aligned.
- Leverage metrics (always shown at bottom):
    Columns: Days, AbsReturn%, CAGR%, SmartScore, CombinedScore, Up/Down, Up%, AvgDailyRange%, DailyMean%, DailyVol%, OptimalLev(x), ZeroLev(x)
"""

import argparse
import sys
import re
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Requires: pip install yfinance pandas python-dateutil", file=sys.stderr); sys.exit(1)

from dateutil.relativedelta import relativedelta


# ---------------- Dataclass ----------------

@dataclass
class Config:
    tickers: list[str]      # any number
    period: str             # 'Nd' | 'Nw' | 'Nm' | 'Ny' | 'ytd'
    basis: str              # 'price' or 'total'
    asof: str | None
    respect_price: bool     # keep raw Close basis even if splits detected
    interval: str           # 'auto' | 'daily' | 'weekly' | 'monthly' | 'yearly'


# ---------------- Window helpers ----------------

def parse_asof(asof: str | None) -> pd.Timestamp:
    if asof:
        return pd.to_datetime(asof).normalize()
    return pd.Timestamp.today().normalize()

def compute_window(asof: pd.Timestamp, period: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """
    Returns (start_dt, end_dt, period_label).
    Supports:
        - 'ytd' (case-insensitive)
        - '<N><unit>' where unit in {d, w, m, y} (days, weeks, months, years)
    Months/years use calendar-accurate relativedelta; days/weeks use timedeltas (float allowed for d/w).
    """
    s = period.strip().lower()
    end_dt = asof

    if s == 'ytd':
        start_dt = pd.Timestamp(year=asof.year, month=1, day=1)
        return start_dt, end_dt, 'YTD'

    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([dwmy])', s)
    if not m:
        raise ValueError("period must be like '10d', '3w', '3m', '2y', or 'ytd'")

    val = float(m.group(1))
    unit = m.group(2)

    if unit == 'd':
        start_dt = asof - pd.Timedelta(days=val)
        label = f"{int(val) if val.is_integer() else val:g}d"
    elif unit == 'w':
        start_dt = asof - pd.Timedelta(days=7 * val)
        label = f"{int(val) if val.is_integer() else val:g}w"
    elif unit == 'm':
        months = int(round(val))
        start_dt = asof - relativedelta(months=months)
        label = f"{months}m"
    else:  # 'y'
        years = int(round(val))
        start_dt = asof - relativedelta(years=years)
        label = f"{years}y"

    return start_dt.normalize(), end_dt.normalize(), label

def fetch_range(ticker: str, start: pd.Timestamp, end: pd.Timestamp, auto_adjust: bool) -> pd.DataFrame:
    # Add small buffers; then clip inclusive [start, end]
    df = yf.download(
        ticker,
        start=start - pd.Timedelta(days=3),
        end=end + pd.Timedelta(days=2),
        interval='1d',  # Keep daily for accurate within-bucket compounding
        auto_adjust=auto_adjust,
        progress=False
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start.date()}..{end.date()}")
    df.index = pd.to_datetime(df.index)
    return df.loc[(df.index >= start) & (df.index <= end)].copy()

def first_trading_on_or_after(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Timestamp:
    candidates = df.index[df.index >= dt]
    if len(candidates) == 0:
        return df.index.min()
    return candidates[0]


# ---------------- Series helpers ----------------

def get_series(df: pd.DataFrame, basis: str) -> pd.Series:
    if basis == 'total' and 'Adj Close' in df.columns:
        s = df['Adj Close']
    else:
        s = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)

def detect_splits_in_window(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    try:
        tk = yf.Ticker(ticker)
        sp = tk.splits
        if sp is None or sp.empty:
            return False
        sp = sp[(sp.index >= start) & (sp.index <= end)]
        return not sp.empty
    except Exception:
        return False


# ---------------- Aggregations ----------------

def daily_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna() * 100.0

def agg_daily_geo(daily_pct: pd.Series) -> pd.DataFrame:
    df = daily_pct.rename('%').to_frame()
    df['trading_days'] = 1
    df['Cum. %'] = ((df['%']/100 + 1).cumprod() - 1) * 100
    df.index.name = 'Date'
    return df[['trading_days','%','Cum. %']]

def agg_weekly_geo(daily_pct: pd.Series) -> pd.DataFrame:
    df = daily_pct.to_frame('daily_pct')
    week_start = (df.index - pd.to_timedelta(df.index.weekday, unit='D')).normalize()
    df['week_start'] = week_start
    grp = df.groupby('week_start', sort=True)
    out = pd.DataFrame({
        'trading_days': grp['daily_pct'].count(),
        '%': grp['daily_pct'].apply(lambda x: ((x/100 + 1).prod() - 1) * 100)
    }).reset_index().sort_values('week_start').reset_index(drop=True)
    out['Cum. %'] = ((out['%']/100 + 1).cumprod() - 1) * 100
    return out.set_index('week_start')[['trading_days','%','Cum. %']]

def agg_monthly_geo(daily_pct: pd.Series) -> pd.DataFrame:
    df = daily_pct.to_frame('daily_pct')
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    grp = df.groupby(['Year','Month'], sort=True)
    out = pd.DataFrame({
        'trading_days': grp['daily_pct'].count(),
        '%': grp['daily_pct'].apply(lambda x: ((x/100 + 1).prod() - 1) * 100)
    }).reset_index().sort_values(['Year','Month']).reset_index(drop=True)
    out['Cum. %'] = ((out['%']/100 + 1).cumprod() - 1) * 100
    return out.set_index(['Year','Month'])[['trading_days','%','Cum. %']]

def agg_yearly_geo(daily_pct: pd.Series) -> pd.DataFrame:
    df = daily_pct.to_frame('daily_pct')
    df['Year'] = df.index.year
    grp = df.groupby('Year', sort=True)
    out = pd.DataFrame({
        'trading_days': grp['daily_pct'].count(),
        '%': grp['daily_pct'].apply(lambda x: ((x/100 + 1).prod() - 1) * 100)
    }).reset_index().sort_values(['Year']).reset_index(drop=True)
    out['Cum. %'] = ((out['%']/100 + 1).cumprod() - 1) * 100
    return out.set_index('Year')[['trading_days','%','Cum. %']]

def simple_return(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan
    return (float(series.iloc[-1]) / float(series.iloc[0]) - 1.0) * 100.0


# ---------------- Pretty ----------------

def format_pct_for_display(df: pd.DataFrame) -> pd.DataFrame:
    disp = df.copy()
    for col in disp.columns:
        if isinstance(col, tuple) and col[1] in ('%', 'Cum. %'):
            # Add thousands separator for large percentages
            disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}%")
    return disp


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Side-by-side return table (calendar-aligned, split-safe).")
    ap.add_argument('--tickers', nargs='+', required=True,
                    help="One or more tickers, e.g., TSLL MSFU MSOX ...")
    ap.add_argument('--period', required=True,
                    help="Flexible: Nd/Nw/Nm/Ny (e.g., 10d, 3w, 3m, 2y) or 'ytd'")
    ap.add_argument('--basis', choices=['price','total'], default='price',
                    help="price=Close; total=Adj Close")
    ap.add_argument('--asof', default=None,
                    help="End date (YYYY-MM-DD). Default: latest close")
    ap.add_argument('--respect-price', action='store_true',
                    help="Do NOT auto-switch to Adj Close even if splits detected")
    ap.add_argument('--interval', choices=['auto','daily','weekly','monthly','yearly'], default='auto',
                    help="Bucket interval for aggregation: auto (>=365d→monthly else weekly), or force daily/weekly/monthly/yearly.")
    ap.add_argument('--explanation', choices=['yes','no'], default='no',
                    help="If 'yes', print one-line explanations of SmartScore, CombinedScore, AvgDailyRange%, DailyMean%, DailyVol% at the end.")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]

    asof_dt = parse_asof(args.asof)
    start_dt, end_dt, per_label = compute_window(asof_dt, args.period)

    # decide aggregation based on --interval
    if args.interval == 'auto':
        monthly_mode = (end_dt - start_dt).days >= 365
        chosen_mode = 'monthly' if monthly_mode else 'weekly'
        forced = False
    else:
        chosen_mode = args.interval
        forced = True

    per_tbl = {}
    diag_rows = []  # for aligned diagnostics table

    # For the leverage table we’ll collect per-ticker metrics here.
    lev_metrics: dict[str, dict] = {}

    for t in tickers:
        # fetch Close (raw) and Adjusted on the SAME calendar window
        df_close = fetch_range(t, start_dt, end_dt, auto_adjust=False)
        start_eff = first_trading_on_or_after(df_close, start_dt)

        had_split = detect_splits_in_window(t, start_dt, end_dt)
        use_total = (args.basis == 'total') or (had_split and not args.respect_price)

        df_used = fetch_range(t, start_dt, end_dt, auto_adjust=use_total)
        s_used  = get_series(df_used, 'total' if use_total else 'price')
        s_used  = s_used.loc[(s_used.index >= start_eff) & (s_used.index <= end_dt)]

        dr = daily_returns(s_used)
        if chosen_mode == 'monthly':
            tbl = agg_monthly_geo(dr)
        elif chosen_mode == 'weekly':
            tbl = agg_weekly_geo(dr)
        elif chosen_mode == 'yearly':
            tbl = agg_yearly_geo(dr)
        else:  # 'daily'
            tbl = agg_daily_geo(dr)

        per_tbl[t] = tbl

        # ---- Diagnostics (collect structured row) ----
        s_close = get_series(df_close, 'price').loc[(df_close.index >= start_eff) & (df_close.index <= end_dt)]
        df_adj  = fetch_range(t, start_dt, end_dt, auto_adjust=True)
        s_adj   = get_series(df_adj, 'total').loc[(df_adj.index >= start_eff) & (df_adj.index <= end_dt)]
        ret_close = simple_return(s_close)
        ret_adj   = simple_return(s_adj)

        auto_switched = (had_split and args.basis == 'price' and not args.respect_price)
        note_str = "Split→Adj" if auto_switched else ("Split" if had_split else "")

        diag_rows.append({
            "ticker": t,
            "start": str(start_eff.date()),
            "end": str(end_dt.date()),
            "ret_close": ret_close,
            "ret_adj": ret_adj,
            "note": note_str,
        })

        # ---- Leverage / summary metrics (always compute) ----
        n_days = int(dr.shape[0])
        pos_cnt = int((dr > 0).sum())
        neg_cnt = int((dr < 0).sum())

        up_pct = (pos_cnt / n_days * 100.0) if n_days > 0 else float('nan')
        avg_abs_pct = float(dr.abs().mean()) if n_days > 0 else float('nan')

        dr_dec = (dr / 100.0)
        mu_dec = float(dr_dec.mean()) if n_days > 0 else float('nan')        # daily mean (decimal)
        e2_dec = float((dr_dec ** 2).mean()) if n_days > 0 else float('nan') # E[r^2]
        var_dec = max(e2_dec - mu_dec**2, 0.0) if n_days > 0 else float('nan')
        sigma_pct = (var_dec ** 0.5) * 100.0 if n_days > 0 else float('nan')
        mu_pct = mu_dec * 100.0 if n_days > 0 else float('nan')

        if n_days > 0 and var_dec > 0:
            L_opt = mu_dec / var_dec
            L_zero = (2.0 * mu_dec) / var_dec
        else:
            L_opt = float('nan')
            L_zero = float('nan')

        # Absolute return and CAGR on the basis actually used in the table (s_used)
        abs_ret_pct = simple_return(s_used)  # % over the window
        if n_days > 0 and not pd.isna(abs_ret_pct):
            cagr_pct = ((1.0 + abs_ret_pct/100.0) ** (252.0 / n_days) - 1.0) * 100.0
        else:
            cagr_pct = float('nan')

        # --- SmartScore (overall edge) ---
        sharpe_like = (mu_pct / sigma_pct) if (not np.isnan(mu_pct) and not np.isnan(sigma_pct) and sigma_pct != 0) else float('nan')
        ratio_range_vol = (avg_abs_pct / sigma_pct) if (not np.isnan(avg_abs_pct) and not np.isnan(sigma_pct) and sigma_pct != 0) else float('nan')
        stability = (1.0 / (1.0 + ratio_range_vol)) if (not np.isnan(ratio_range_vol)) else float('nan')
        edge_score = sharpe_like * stability * 100.0 if (not np.isnan(sharpe_like) and not np.isnan(stability)) else float('nan')

        # --- CombinedScore = CAGR% × SmartScore ---
        combined_score = (cagr_pct * edge_score) if (not np.isnan(cagr_pct) and not np.isnan(edge_score)) else float('nan')

        lev_metrics[t] = {
            'days': n_days,
            'abs_ret_pct': abs_ret_pct,
            'cagr_pct': cagr_pct,
            'edge_score': edge_score,
            'combined_score': combined_score,
            'pos': pos_cnt,
            'neg': neg_cnt,
            'up_pct': up_pct,
            'avg_abs_pct': avg_abs_pct,
            'mu_pct': mu_pct,
            'sigma_pct': sigma_pct,
            'L_opt': L_opt,
            'L_zero': L_zero,
        }

    # align indexes across tickers for the side-by-side table
    index_union = None
    for t, tbl in per_tbl.items():
        index_union = tbl.index if index_union is None else index_union.union(tbl.index)
    for t in tickers:
        per_tbl[t] = per_tbl[t].reindex(index_union)

    # Single "Trading days" column (max across tickers)
    td_df = pd.concat({t: per_tbl[t]['trading_days'] for t in tickers}, axis=1)
    trading_days_single = td_df.max(axis=1)

    # Drop per-ticker 'trading_days' and merge metrics columns
    for t in tickers:
        per_tbl[t] = per_tbl[t].drop(columns=['trading_days'])
    sbs = pd.concat(per_tbl, axis=1)

    # Reset index and insert Trading days
    result = sbs.reset_index()

    # Identify date index columns
    if 'week_start' in result.columns:
        index_cols = ['week_start']
    elif all(col in result.columns for col in ['Year', 'Month']):
        index_cols = ['Year', 'Month']
    elif 'Year' in result.columns:
        index_cols = ['Year']
    elif 'Date' in result.columns:
        index_cols = ['Date']
    else:
        index_cols = [result.columns[0]]

    insert_pos = len(index_cols)
    result.insert(insert_pos, 'Trading days', trading_days_single.values)

    # Pretty-format % columns (with thousands separators)
    disp = result.copy()
    for col in disp.columns:
        if isinstance(col, tuple) and col[1] in ('%', 'Cum. %'):
            disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}%")

    # Header
    mode_human = {
        'daily': 'DAILY',
        'weekly': 'WEEKLY (Monday-anchored)',
        'monthly': 'MONTHLY (calendar months)',
        'yearly': 'YEARLY (calendar years)',
    }[chosen_mode]
    basis_note = " (auto-switch Adj on splits)" if (args.basis == 'price' and not args.respect_price) else ""
    forced_tag = " [forced]" if forced else ""
    header = (
        f"\nTickers: {', '.join(tickers)} | Period: {per_label} | Basis: {args.basis.upper()}{basis_note} | "
        f"Window: {start_dt.date()}→{end_dt.date()} | Mode: {mode_human}{forced_tag}"
    )
    print(header)
    print("-" * len(header))
    print(disp.to_string(index=False))

    # ----- Leverage / summary metrics (always shown) -----
    print("\n— Leverage metrics —")
    print(f"Window: {start_dt.date()}→{end_dt.date()} | Mode: {mode_human}")
    w_tk = max(6, max(len(t) for t in tickers))
    # Ticker  Days  AbsReturn%  CAGR%  SmartScore  CombinedScore  Up/Down  Up%  AvgDailyRange%  DailyMean%  DailyVol%  OptimalLev(x)  ZeroLev(x)
    header_fmt = (
        f"{{:<{w_tk}}}  {{:>6}}  {{:>12}}  {{:>8}}  {{:>10}}  {{:>14}}  {{:>9}}  {{:>7}}  "
        f"{{:>16}}  {{:>11}}  {{:>10}}  {{:>13}}  {{:>11}}"
    )
    # NOTE: add thousands separators ONLY for percentage fields
    line_fmt = (
        f"{{:<{w_tk}}}  {{:>6d}}  {{:>12,.2f}}  {{:>8,.2f}}  {{:>10.2f}}  {{:>14.2f}}  {{:>9}}  {{:>7,.2f}}  "
        f"{{:>16,.2f}}  {{:>11,.2f}}  {{:>10,.2f}}  {{:>13.2f}}  {{:>11.2f}}"
    )
    print(header_fmt.format(
        "Ticker", "Days", "AbsReturn%", "CAGR%", "SmartScore", "CombinedScore", "Up/Down", "Up%",
        "AvgDailyRange%", "DailyMean%", "DailyVol%", "OptimalLev(x)", "ZeroLev(x)"
    ))

    # Build rows, sort by CombinedScore (desc), then print
    rows = []
    for t in tickers:
        m = lev_metrics[t]
        updown = f"{m['pos']}/{m['neg']}"
        rows.append((
            t, m['days'], m['abs_ret_pct'], m['cagr_pct'], m['edge_score'], m['combined_score'],
            updown, m['up_pct'], m['avg_abs_pct'], m['mu_pct'], m['sigma_pct'], m['L_opt'], m['L_zero']
        ))
    rows.sort(key=lambda x: (float('-inf') if pd.isna(x[5]) else x[5]), reverse=True)

    for r in rows:
        print(line_fmt.format(*r))

    # ----- Optional one-line explanations -----
    if args.explanation == 'yes':
        print("\n— Explanations —")
        print("SmartScore: Higher is better; ≈ (DailyMean / DailyVol) × 1/(1 + AvgDailyRange/DailyVol) × 100, rewarding drift and penalizing volatility/intraday chop.")
        print("CombinedScore: CAGR × SmartScore; balances raw growth with efficiency so high-but-fragile names don't dominate.")
        print("AvgDailyRange%: Average intraday (High–Low)/Open; higher = choppier candles and more intraday noise.")
        print("DailyMean%: Average close-to-close return per trading day; higher = stronger positive drift.")
        print("DailyVol%: Standard deviation of daily returns; lower = smoother path and higher safe leverage.")

# ---------------- CSV helpers ----------------

def _flatten_columns_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns like (TICKER, '%') -> 'TICKER %' for CSV."""
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(" ".join(str(x) for x in c if x != ""))
        else:
            new_cols.append(str(c))
    out = df.copy()
    out.columns = new_cols
    return out

if __name__ == "__main__":
    # Extend CLI to support CSV export
    def patched_main():
        ap = argparse.ArgumentParser(description="Side-by-side return table (calendar-aligned, split-safe).")
        ap.add_argument('--tickers', nargs='+', required=True,
                        help="One or more tickers, e.g., TSLL MSFU MSOX ...")
        ap.add_argument('--period', required=True,
                        help="Flexible: Nd/Nw/Nm/Ny (e.g., 10d, 3w, 3m, 2y) or 'ytd'")
        ap.add_argument('--basis', choices=['price','total'], default='price',
                        help="price=Close; total=Adj Close")
        ap.add_argument('--asof', default=None,
                        help="End date (YYYY-MM-DD). Default: latest close")
        ap.add_argument('--respect-price', action='store_true',
                        help="Do NOT auto-switch to Adj Close even if splits detected")
        ap.add_argument('--interval', choices=['auto','daily','weekly','monthly','yearly'], default='auto',
                        help="Bucket interval for aggregation: auto (>=365d→monthly else weekly), or force daily/weekly/monthly/yearly.")
        ap.add_argument('--explanation', choices=['yes','no'], default='no',
                        help="If 'yes', print one-line explanations of SmartScore, CombinedScore, AvgDailyRange%, DailyMean%, DailyVol% at the end.")
        # NEW: CSV outputs
        ap.add_argument('--csv-out', default=None,
                        help="Path to save the side-by-side table as CSV (numeric, not pretty-formatted).")
        ap.add_argument('--csv-metrics-out', default=None,
                        help="Path to save the leverage metrics table as CSV.")

        args = ap.parse_args()

        # ---- below here is the original main() logic, minimally adapted to use 'args' ----
        tickers = [t.upper() for t in args.tickers]

        asof_dt = parse_asof(args.asof)
        start_dt, end_dt, per_label = compute_window(asof_dt, args.period)

        # decide aggregation based on --interval
        if args.interval == 'auto':
            monthly_mode = (end_dt - start_dt).days >= 365
            chosen_mode = 'monthly' if monthly_mode else 'weekly'
            forced = False
        else:
            chosen_mode = args.interval
            forced = True

        per_tbl = {}
        diag_rows = []  # for aligned diagnostics table

        # For the leverage table we’ll collect per-ticker metrics here.
        lev_metrics: dict[str, dict] = {}

        for t in tickers:
            # fetch Close (raw) and Adjusted on the SAME calendar window
            df_close = fetch_range(t, start_dt, end_dt, auto_adjust=False)
            start_eff = first_trading_on_or_after(df_close, start_dt)

            had_split = detect_splits_in_window(t, start_dt, end_dt)
            use_total = (args.basis == 'total') or (had_split and not args.respect_price)

            df_used = fetch_range(t, start_dt, end_dt, auto_adjust=use_total)
            s_used  = get_series(df_used, 'total' if use_total else 'price')
            s_used  = s_used.loc[(s_used.index >= start_eff) & (s_used.index <= end_dt)]

            dr = daily_returns(s_used)
            if chosen_mode == 'monthly':
                tbl = agg_monthly_geo(dr)
            elif chosen_mode == 'weekly':
                tbl = agg_weekly_geo(dr)
            elif chosen_mode == 'yearly':
                tbl = agg_yearly_geo(dr)
            else:  # 'daily'
                tbl = agg_daily_geo(dr)

            per_tbl[t] = tbl

            # ---- Diagnostics (collect structured row) ----
            s_close = get_series(df_close, 'price').loc[(df_close.index >= start_eff) & (df_close.index <= end_dt)]
            df_adj  = fetch_range(t, start_dt, end_dt, auto_adjust=True)
            s_adj   = get_series(df_adj, 'total').loc[(df_adj.index >= start_eff) & (df_adj.index <= end_dt)]
            ret_close = simple_return(s_close)
            ret_adj   = simple_return(s_adj)

            auto_switched = (had_split and args.basis == 'price' and not args.respect_price)
            note_str = "Split→Adj" if auto_switched else ("Split" if had_split else "")

            diag_rows.append({
                "ticker": t,
                "start": str(start_eff.date()),
                "end": str(end_dt.date()),
                "ret_close": ret_close,
                "ret_adj": ret_adj,
                "note": note_str,
            })

            # ---- Leverage / summary metrics (always compute) ----
            n_days = int(dr.shape[0])
            pos_cnt = int((dr > 0).sum())
            neg_cnt = int((dr < 0).sum())

            up_pct = (pos_cnt / n_days * 100.0) if n_days > 0 else float('nan')
            avg_abs_pct = float(dr.abs().mean()) if n_days > 0 else float('nan')

            dr_dec = (dr / 100.0)
            mu_dec = float(dr_dec.mean()) if n_days > 0 else float('nan')        # daily mean (decimal)
            e2_dec = float((dr_dec ** 2).mean()) if n_days > 0 else float('nan') # E[r^2]
            var_dec = max(e2_dec - mu_dec**2, 0.0) if n_days > 0 else float('nan')
            sigma_pct = (var_dec ** 0.5) * 100.0 if n_days > 0 else float('nan')
            mu_pct = mu_dec * 100.0 if n_days > 0 else float('nan')

            if n_days > 0 and var_dec > 0:
                L_opt = mu_dec / var_dec
                L_zero = (2.0 * mu_dec) / var_dec
            else:
                L_opt = float('nan')
                L_zero = float('nan')

            # Absolute return and CAGR on the basis actually used in the table (s_used)
            abs_ret_pct = simple_return(s_used)  # % over the window
            if n_days > 0 and not pd.isna(abs_ret_pct):
                cagr_pct = ((1.0 + abs_ret_pct/100.0) ** (252.0 / n_days) - 1.0) * 100.0
            else:
                cagr_pct = float('nan')

            # --- SmartScore (overall edge) ---
            sharpe_like = (mu_pct / sigma_pct) if (not np.isnan(mu_pct) and not np.isnan(sigma_pct) and sigma_pct != 0) else float('nan')
            ratio_range_vol = (avg_abs_pct / sigma_pct) if (not np.isnan(avg_abs_pct) and not np.isnan(sigma_pct) and sigma_pct != 0) else float('nan')
            stability = (1.0 / (1.0 + ratio_range_vol)) if (not np.isnan(ratio_range_vol)) else float('nan')
            edge_score = sharpe_like * stability * 100.0 if (not np.isnan(sharpe_like) and not np.isnan(stability)) else float('nan')

            # --- CombinedScore = CAGR% × SmartScore ---
            combined_score = (cagr_pct * edge_score) if (not np.isnan(cagr_pct) and not np.isnan(edge_score)) else float('nan')

            lev_metrics[t] = {
                'Ticker': t,
                'Days': n_days,
                'AbsReturn%': abs_ret_pct,
                'CAGR%': cagr_pct,
                'SmartScore': edge_score,
                'CombinedScore': combined_score,
                'Up/Down': f"{pos_cnt}/{neg_cnt}",
                'Up%': up_pct,
                'AvgDailyRange%': avg_abs_pct,
                'DailyMean%': mu_pct,
                'DailyVol%': sigma_pct,
                'OptimalLev(x)': L_opt,
                'ZeroLev(x)': L_zero,
            }

        # align indexes across tickers for the side-by-side table
        index_union = None
        for t, tbl in per_tbl.items():
            index_union = tbl.index if index_union is None else index_union.union(tbl.index)
        for t in tickers:
            per_tbl[t] = per_tbl[t].reindex(index_union)

        # Single "Trading days" column (max across tickers)
        td_df = pd.concat({t: per_tbl[t]['trading_days'] for t in tickers}, axis=1)
        trading_days_single = td_df.max(axis=1)

        # Drop per-ticker 'trading_days' and merge metrics columns
        for t in tickers:
            per_tbl[t] = per_tbl[t].drop(columns=['trading_days'])
        sbs = pd.concat(per_tbl, axis=1)

        # Reset index and insert Trading days
        result = sbs.reset_index()

        # Identify date index columns
        if 'week_start' in result.columns:
            index_cols = ['week_start']
        elif all(col in result.columns for col in ['Year', 'Month']):
            index_cols = ['Year', 'Month']
        elif 'Year' in result.columns:
            index_cols = ['Year']
        elif 'Date' in result.columns:
            index_cols = ['Date']
        else:
            index_cols = [result.columns[0]]

        insert_pos = len(index_cols)
        result.insert(insert_pos, 'Trading days', trading_days_single.values)

        # OPTIONAL: write numeric CSV for the side-by-side table
        if args.csv_out:
            csv_df = _flatten_columns_for_csv(result)
            csv_df.to_csv(args.csv_out, index=False, encoding='utf-8-sig')

        # Build leverage metrics DataFrame (numeric) and optional CSV
        metrics_df = pd.DataFrame(list(lev_metrics.values()))
        if not metrics_df.empty:
            # sort by CombinedScore desc (like the printed table)
            metrics_df = metrics_df.sort_values('CombinedScore', ascending=False)
        if args.csv_metrics_out:
            metrics_df.to_csv(args.csv_metrics_out, index=False, encoding='utf-8-sig')

        # ---- pretty print (unchanged from original main) ----
        # Pretty-format % columns (with thousands separators)
        disp = result.copy()
        for col in disp.columns:
            if isinstance(col, tuple) and col[1] in ('%', 'Cum. %'):
                disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}%")

        mode_human = {
            'daily': 'DAILY',
            'weekly': 'WEEKLY (Monday-anchored)',
            'monthly': 'MONTHLY (calendar months)',
            'yearly': 'YEARLY (calendar years)',
        }[chosen_mode]
        basis_note = " (auto-switch Adj on splits)" if (args.basis == 'price' and not args.respect_price) else ""
        forced_tag = " [forced]" if forced else ""
        header = (
            f"Tickers: {', '.join(tickers)} | Period: {per_label} | Basis: {args.basis.upper()}{basis_note} | "
            f"Window: {start_dt.date()}→{end_dt.date()} | Mode: {mode_human}{forced_tag}"
        )
        print(header)
        print("-" * len(header))
        print(disp.to_string(index=False))

        print("— Leverage metrics —")
        print(f"Window: {start_dt.date()}→{end_dt.date()} | Mode: {mode_human}")
        w_tk = max(6, max(len(t) for t in tickers))
        header_fmt = (
            f"{{:<{w_tk}}}  {{:>6}}  {{:>12}}  {{:>8}}  {{:>10}}  {{:>14}}  {{:>9}}  {{:>7}}  "
            f"{{:>16}}  {{:>11}}  {{:>10}}  {{:>13}}  {{:>11}}"
        )
        line_fmt = (
            f"{{:<{w_tk}}}  {{:>6d}}  {{:>12,.2f}}  {{:>8,.2f}}  {{:>10.2f}}  {{:>14.2f}}  {{:>9}}  {{:>7,.2f}}  "
            f"{{:>16,.2f}}  {{:>11,.2f}}  {{:>10,.2f}}  {{:>13.2f}}  {{:>11.2f}}"
        )
        print(header_fmt.format(
            "Ticker", "Days", "AbsReturn%", "CAGR%", "SmartScore", "CombinedScore", "Up/Down", "Up%",
            "AvgDailyRange%", "DailyMean%", "DailyVol%", "OptimalLev(x)", "ZeroLev(x)"
        ))

        rows = []
        for t in tickers:
            m = lev_metrics[t]
            updown = m['Up/Down']
            rows.append((
                t, m['Days'], m['AbsReturn%'], m['CAGR%'], m['SmartScore'], m['CombinedScore'],
                updown, m['Up%'], m['AvgDailyRange%'], m['DailyMean%'], m['DailyVol%'], m['OptimalLev(x)'], m['ZeroLev(x)']
            ))
        rows.sort(key=lambda x: (float('-inf') if pd.isna(x[5]) else x[5]), reverse=True)

        for r in rows:
            print(line_fmt.format(*r))

        if args.explanation == 'yes':
            print("— Explanations —")
            print("SmartScore: Higher is better; ≈ (DailyMean / DailyVol) × 1/(1 + AvgDailyRange/DailyVol) × 100, rewarding drift and penalizing volatility/intraday chop.")
            print("CombinedScore: CAGR × SmartScore; balances raw growth with efficiency so high-but-fragile names don't dominate.")
            print("AvgDailyRange%: Average intraday (High–Low)/Open; higher = choppier candles and more intraday noise.")
            print("DailyMean%: Average close-to-close return per trading day; higher = stronger positive drift.")
            print("DailyVol%: Standard deviation of daily returns; lower = smoother path and higher safe leverage.")

    patched_main()
