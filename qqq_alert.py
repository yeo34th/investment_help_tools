import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import yfinance as yf

# TICKER = "QQQ"
# SENDER = os.getenv("GMAIL_USER")
# APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
# RECIPIENT = os.getenv("RECIPIENT_EMAIL", SENDER)

TICKER = "QQQ"
SENDER = "jerichoconsultingllc@gmail.com"
APP_PASSWORD = "wfyx mncu jltt snfl"
RECIPIENT = "changmo.yeo1@gmail.com"

def fetch_metrics():
    df = yf.Ticker(TICKER).history(period="max", auto_adjust=False)
    if df.empty or "Close" not in df:
        raise RuntimeError("No price data for QQQ.")
    close = df["Close"].dropna()
    last = float(close.iloc[-1])

    ath = float(close.max())
    ath_date = close.idxmax()
    prev_ath = float(close.iloc[:-1].max()) if len(close) > 1 else ath

    drawdown = max(0.0, (1 - last / ath) * 100.0)

    is_new_ath = (ath_date == close.index[-1]) and (ath > prev_ath + 1e-9)
    ath_change_abs = ath - prev_ath
    ath_change_pct = (ath_change_abs / prev_ath * 100.0) if prev_ath > 0 else 0.0

    return {
        "last": last,
        "ath": ath,
        "ath_date": ath_date,
        "prev_ath": prev_ath,
        "drawdown": drawdown,
        "is_new_ath": is_new_ath,
        "ath_change_abs": ath_change_abs,
        "ath_change_pct": ath_change_pct,
    }


def band_and_action(dd: float) -> tuple[str, str]:
    """
    Map drawdown percentage to your deployment/rebalance plan.
    Ranges: [lower, upper)
    """
    if dd == 0:
        return ("🌟 NEW ATH", "New all-time high today.")
    if dd < 15:
        return ("Monitoring (pre-deploy)", "No deployment. Wait for ≥15% drawdown.")
    if 15 <= dd < 25:
        return ("1st deployment window (−15% to −25%)", "Execute 1st deployment tranche.")
    if 25 <= dd < 35:
        return ("2nd deployment window (−25% to −35%)", "Execute 2nd deployment tranche.")
    if 35 <= dd < 45:
        return ("3rd deployment window (−35% to −45%)", "Execute 3rd deployment tranche.")
    if 45 <= dd < 55:
        return ("Rebalance: Convert half QQQ→TQQQ (−45% to −55%)",
                "Convert 50% of remaining QQQ into TQQQ.")
    if 55 <= dd < 65:
        return ("Rebalance: Convert remaining QQQ→TQQQ (−55% to −65%)",
                "Convert the rest of QQQ into TQQQ.")
    return ("Max deployment reached (≥−65%)",
            "All tranches used; conversions complete. Hold to recovery per plan.")


def fetch_macro_indicators():
    """
    Get VIX, 10Y yield, DXY and build a combined Market mode line.
    Output format target:
      VIX: <num>
      10Y Yield: <num>%
      DXY: <num>
      Market mode: <base> · <vol phrase> · <rate phrase>, <dollar phrase>
    """
    symbols = {
        "VIX": "^VIX",
        "TNX": "^TNX",
        "DXY1": "DX-Y.NYB",
        "DXY2": "DXY",
    }

    def last_close(ticker):
        try:
            df = yf.Ticker(ticker).history(period="5d")
            if df.empty:
                return None
            return float(df["Close"].dropna().iloc[-1])
        except Exception:
            return None

    vix = last_close(symbols["VIX"])
    tnx_raw = last_close(symbols["TNX"])
    dxy = last_close(symbols["DXY1"]) or last_close(symbols["DXY2"])

    # ---- 10Y Yield scaling ----
    if tnx_raw is None:
        ten_y = 0.0
    else:
        if tnx_raw > 20:
            ten_y = tnx_raw / 10.0      # 42.7 -> 4.27%
        elif tnx_raw > 1:
            ten_y = tnx_raw             # already %
        else:
            ten_y = tnx_raw * 100.0     # 0.042 -> 4.2%

    vix = vix if vix is not None else 0.0
    dxy = dxy if dxy is not None else 0.0

    # ---- classify states ----
    # Volatility state
    if vix == 0.0:
        vix_state = "N/A"
    elif vix < 15:
        vix_state = "Calm"
    elif vix < 25:
        vix_state = "Moderate"
    else:
        vix_state = "High"

    # Rate state
    if ten_y == 0.0:
        rate_state = "N/A"
    elif ten_y < 3.5:
        rate_state = "Low"
    elif ten_y < 4.5:
        rate_state = "Neutral"
    else:
        rate_state = "Rising"

    # Dollar strength
    if dxy == 0.0:
        usd_state = "N/A"
    elif dxy < 100:
        usd_state = "Weak"
    elif dxy < 105:
        usd_state = "Neutral"
    else:
        usd_state = "Strong"

    # phrases
    if vix_state == "Calm":
        vol_phrase = "Calm volatility"
    elif vix_state == "Moderate":
        vol_phrase = "Moderate volatility"
    elif vix_state == "High":
        vol_phrase = "High volatility"
    else:
        vol_phrase = "N/A volatility"

    if rate_state == "Low":
        rate_phrase = "Low rates"
    elif rate_state == "Neutral":
        rate_phrase = "Neutral rates"
    elif rate_state == "Rising":
        rate_phrase = "Rising rates"
    else:
        rate_phrase = "N/A rates"

    if usd_state == "Weak":
        usd_phrase = "Weak dollar"
    elif usd_state == "Neutral":
        usd_phrase = "Neutral dollar"
    elif usd_state == "Strong":
        usd_phrase = "Strong dollar"
    else:
        usd_phrase = "N/A dollar"

    # base mode
    if vix_state == "High":
        base_mode = "Risk-off tone"
    elif rate_state == "Rising" and usd_state == "Strong":
        base_mode = "Tight conditions"
    elif rate_state == "Rising":
        base_mode = "Rate pressure"
    elif usd_state == "Strong":
        base_mode = "Strong dollar backdrop"
    elif vix_state == "Calm" and rate_state in ("Low", "Neutral"):
        base_mode = "Stable momentum"
    else:
        base_mode = "Mixed signals"

    market_mode = f"{base_mode} · {vol_phrase} · {rate_phrase}, {usd_phrase}"

    return {
        "VIX": f"{vix:.1f}" if vix > 0 else "N/A",
        "10Y": f"{ten_y:.2f}%" if ten_y > 0 else "N/A",
        "DXY": f"{dxy:.1f}" if dxy > 0 else "N/A",
        "Market": market_mode,
    }


def send_email(subject, body):
    msg = EmailMessage()
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)


if __name__ == "__main__":
    m = fetch_metrics()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    band_label, action_line = band_and_action(m["drawdown"])

    if m["is_new_ath"]:
        subject = (
            f"🌟 NEW ATH: QQQ ${m['ath']:.2f} "
            f"(+{m['ath_change_abs']:.2f}, {m['ath_change_pct']:.2f}%)"
        )
    else:
        subject = f"{band_label} · QQQ {m['drawdown']:.2f}% below ATH"

    body = (
        f"QQQ daily check ({now})\n\n"
        f"Current Price: ${m['last']:.2f}\n"
        f"All-Time High: ${m['ath']:.2f} (set {m['ath_date'].date()})\n"
        f"Drawdown from ATH: {m['drawdown']:.2f}%\n"
        f"Status: {band_label}\n"
        f"Action: {action_line}\n"
    )

    if m["is_new_ath"]:
        body += (
            f"\n— New ATH today —\n"
            f"Previous ATH: ${m['prev_ath']:.2f}\n"
            f"ATH Change: +${m['ath_change_abs']:.2f} "
            f"({m['ath_change_pct']:.2f}%)\n"
        )

    macro = fetch_macro_indicators()
    body += (
        "\n"
        f"VIX: {macro['VIX']}\n"
        f"10Y Yield: {macro['10Y']}\n"
        f"DXY: {macro['DXY']}\n"
        f"Market mode: {macro['Market']}\n"
    )

    send_email(subject, body)
    print(f"Sent: {subject}")