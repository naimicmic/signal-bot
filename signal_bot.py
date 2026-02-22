import os
import time
import requests
import ccxt

# =========================
# CONFIG
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

TIMEFRAME_MAIN = "4h"          # <-- schimbat din 1h in 4h
SCAN_EVERY_MINUTES = 30        # recomandat pentru 4h (poți pune 60)

# Burse: adaugat OKX + Bybit
EXCHANGES = ["binance", "kucoin", "gate", "exmo", "okx", "bybit"]

SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","HBAR/USDT",
    "AVAX/USDT","LINK/USDT","XRP/USDT","MATIC/USDT","SUI/USDT","SEI/USDT",
    "HYPE/USDT","ONDO/USDT"
]

# TP / risk
TP1_PCT = 0.02
TP2_PCT = 0.04
TP3_PCT = 0.06
TP3_EXT_PCT = 0.10

MIN_EXCH_CONFIRM = 2          # semnal valid daca apare pe >= 2 burse
CANDLES_LIMIT = 260           # suficient pt EMA200 + MACD + StochRSI
MIN_SCORE_TO_ALERT = 8.0      # trimite doar semnale "tare"

# =========================
# INDICATORS
# =========================
def ema_series(values, period):
    """Returnează seria EMA (listă), aceeași lungime ca input (primele pot fi aproximative)."""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    out = [values[0]]
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def ema_last(values, period):
    s = ema_series(values, period)
    return s[-1] if s else None

def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += -diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        r = 100.0
    else:
        rs = avg_gain / avg_loss
        r = 100 - (100 / (1 + rs))
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            r = 100.0
        else:
            rs = avg_gain / avg_loss
            r = 100 - (100 / (1 + rs))
    return r

def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a

def swing_low(lows, lookback=12):
    return min(lows[-lookback:]) if len(lows) >= lookback else (min(lows) if lows else None)

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def stoch_rsi(values, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Returnează (k, d) pentru StochRSI.
    """
    if len(values) < rsi_period + stoch_period + 10:
        return (None, None)

    # calc RSI series
    rsi_vals = []
    for i in range(rsi_period, len(values)):
        r = rsi(values[: i + 1], rsi_period)
        rsi_vals.append(r)

    if len(rsi_vals) < stoch_period + d_period:
        return (None, None)

    # StochRSI raw
    stoch_raw = []
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1 : i + 1]
        lo = min(window)
        hi = max(window)
        if hi == lo:
            stoch_raw.append(0.0)
        else:
            stoch_raw.append((rsi_vals[i] - lo) / (hi - lo) * 100.0)

    if len(stoch_raw) < max(k_period, d_period) + 2:
        return (None, None)

    # %K = SMA(stoch_raw, k_period), %D = SMA(%K, d_period)
    k_series = []
    for i in range(k_period - 1, len(stoch_raw)):
        k_series.append(sum(stoch_raw[i - k_period + 1 : i + 1]) / k_period)

    if len(k_series) < d_period:
        return (None, None)

    d_val = sum(k_series[-d_period:]) / d_period
    k_val = k_series[-1]
    return (k_val, d_val)

def macd(values, fast=12, slow=26, signal=9):
    """
    Returnează (macd_line, signal_line, hist)
    """
    if len(values) < slow + signal + 10:
        return (None, None, None)

    ema_fast = ema_series(values, fast)
    ema_slow = ema_series(values, slow)
    if not ema_fast or not ema_slow:
        return (None, None, None)

    # aliniem seriile (au aceeași lungime)
    macd_line_series = [f - s for f, s in zip(ema_fast, ema_slow)]

    signal_series = ema_series(macd_line_series, signal)
    if not signal_series:
        return (None, None, None)

    macd_line = macd_line_series[-1]
    signal_line = signal_series[-1]
    hist = macd_line - signal_line
    return (macd_line, signal_line, hist)

# =========================
# TELEGRAM
# =========================
def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Lipseste TELEGRAM_TOKEN sau TELEGRAM_CHAT_ID.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=20)
    if not r.ok:
        print("Eroare Telegram:", r.text)

# =========================
# EXCHANGE INIT
# =========================
def init_exchange(name):
    config = {"enableRateLimit": True}

    # Forțează SPOT acolo unde e nevoie
    if name in ["bybit", "okx"]:
        config["options"] = {"defaultType": "spot"}

    ex = getattr(ccxt, name)(config)
    ex.load_markets()
    return ex

def fetch_ohlcv_safe(ex, symbol, timeframe, limit):
    if symbol not in ex.markets:
        return None
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return None

# =========================
# SIGNAL LOGIC (EMA + RSI + ATR + VOL + MACD + StochRSI)
# =========================
def analyze_symbol(ohlcv):
    if not ohlcv or len(ohlcv) < 210:
        return None

    closes = [c[4] for c in ohlcv]
    highs  = [c[2] for c in ohlcv]
    lows   = [c[3] for c in ohlcv]
    vols   = [c[5] for c in ohlcv]

    last_close = closes[-1]

    # Core indicators
    ema20  = ema_last(closes[-220:], 20)
    ema50  = ema_last(closes[-240:], 50)
    ema200 = ema_last(closes[-260:], 200)
    r = rsi(closes[-220:], 14)
    a = atr(highs[-220:], lows[-220:], closes[-220:], 14)

    if None in (ema20, ema50, ema200, r, a):
        return None

    # Volume soft check
    v20 = sum(vols[-20:]) / 20 if len(vols) >= 20 else vols[-1]
    vol_ok = vols[-1] >= 0.9 * v20

    # MACD
    macd_line, signal_line, hist = macd(closes[-260:], 12, 26, 9)
    if None in (macd_line, signal_line, hist):
        return None
    macd_bull = (macd_line > signal_line) and (hist > 0)
    macd_bear = (macd_line < signal_line) and (hist < 0)

    # StochRSI
    k, d = stoch_rsi(closes[-260:], 14, 14, 3, 3)
    if None in (k, d):
        return None
    stoch_cross_up = (k > d) and (k < 30)    # trigger din zona joasă
    stoch_cross_dn = (k < d) and (k > 70)    # avertizare din zona sus

    # Trend filter (4h)
    trend_bull = (ema50 > ema200) and (last_close > ema200) and (ema20 > ema50)
    trend_weak = (ema50 < ema200) or (last_close < ema200)

    # Pullback near EMA
    near_ema = (abs(last_close - ema20) / ema20 < 0.007) or (abs(last_close - ema50) / ema50 < 0.012)

    # RSI reclaim
    rsi_ok_buy = r >= 50.0
    rsi_sell = r < 45.0

    # BUY logic: trend + pullback + momentum confirm (MACD) + timing (StochRSI) + volume
    buy = trend_bull and near_ema and rsi_ok_buy and macd_bull and (stoch_cross_up or k < 35) and vol_ok

    # SELL warning: trend breaks OR momentum bearish (MACD) + RSI weak + price under EMA50 (sau stoch cross down)
    sell = ((trend_weak and macd_bear) or (last_close < ema50 and rsi_sell and macd_bear) or stoch_cross_dn) and vol_ok

    # SL via swing low + ATR
    entry = last_close
    sl1 = swing_low(lows, lookback=12)
    sl2 = entry - 1.5 * a
    sl = min(sl1, sl2) if sl1 is not None else sl2

    # TPs
    tp1 = entry * (1 + TP1_PCT)
    tp2 = entry * (1 + TP2_PCT)
    tp3 = entry * (1 + TP3_PCT)

    strong_trend = (last_close > ema200 * 1.02) and (r >= 55.0) and (hist > 0)
    tp3_ext = entry * (1 + TP3_EXT_PCT) if strong_trend else None

    # Score 0..10 (adaptat)
    score = 0.0
    if trend_bull: score += 3.5
    if near_ema: score += 1.5
    if r >= 55: score += 1.0
    if vol_ok: score += 1.0
    if macd_bull: score += 2.0
    if stoch_cross_up: score += 1.0
    if strong_trend: score += 1.0
    score = min(score, 10.0)

    reasons = [
        f"EMA50>EMA200={ema50>ema200}",
        f"Close>EMA200={last_close>ema200}",
        f"EMA20>EMA50={ema20>ema50}",
        f"RSI={r:.1f}",
        f"MACD={'BULL' if macd_bull else ('BEAR' if macd_bear else 'NEUTRAL')}",
        f"StochRSI K/D={k:.1f}/{d:.1f}",
        f"VolOK={vol_ok}",
        f"TF={TIMEFRAME_MAIN}",
    ]

    return {
        "buy": buy,
        "sell": sell,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp3_ext": tp3_ext,
        "score": score,
        "reason": "; ".join(reasons),
    }

def format_signal(symbol, kind, exchanges, info):
    entry = info["entry"]
    sl = info["sl"]
    risk_pct = (entry - sl) / entry * 100 if entry and sl else None

    lines = []
    lines.append(f"<b>{kind} (SPOT)</b>  <b>{symbol}</b>")
    lines.append(f"Confirmare: {', '.join(exchanges)} (>= {MIN_EXCH_CONFIRM})")
    lines.append(f"TF: {TIMEFRAME_MAIN}")
    lines.append("")
    lines.append(f"Entry: <b>{entry:.6g}</b>")
    lines.append(f"SL: <b>{sl:.6g}</b>  ({risk_pct:.2f}%)" if risk_pct is not None else f"SL: <b>{sl:.6g}</b>")

    if kind == "BUY":
        lines.append(f"TP1: {info['tp1']:.6g} (+{TP1_PCT*100:.0f}%) | ~40%")
        lines.append(f"TP2: {info['tp2']:.6g} (+{TP2_PCT*100:.0f}%) | ~35%")
        lines.append(f"TP3: {info['tp3']:.6g} (+{TP3_PCT*100:.0f}%) | ~15%")
        if info["tp3_ext"]:
            lines.append(f"TP3 EXT: {info['tp3_ext']:.6g} (+{TP3_EXT_PCT*100:.0f}%) | runner ~10% (trend strong)")
        else:
            lines.append("TP3 EXT: off -> recomand trailing dupa TP2")
    else:
        lines.append("SELL = avertizare (posibila cadere / iesire partiala).")

    lines.append("")
    lines.append(f"Scor: <b>{info['score']:.1f}/10</b> (min {MIN_SCORE_TO_ALERT:.1f})")
    lines.append(f"Motiv: {info['reason']}")
    return "\n".join(lines)

def main_loop():
    ex_objs = {}
    for name in EXCHANGES:
        try:
            ex_objs[name] = init_exchange(name)
            print("Init OK:", name)
        except Exception as e:
            print("Init esuat:", name, e)

    last_sent = set()  # dedupe simplu

    while True:
        for symbol in SYMBOLS:
            buys = []
            sells = []
            per_ex_info = {}

            for name, ex in ex_objs.items():
                ohlcv = fetch_ohlcv_safe(ex, symbol, TIMEFRAME_MAIN, CANDLES_LIMIT)
                info = analyze_symbol(ohlcv)
                if not info:
                    continue

                per_ex_info[name] = info
                if info["buy"] and info["score"] >= MIN_SCORE_TO_ALERT:
                    buys.append(name)
                if info["sell"] and info["score"] >= MIN_SCORE_TO_ALERT:
                    sells.append(name)

            # BUY confirm
            if len(buys) >= MIN_EXCH_CONFIRM:
                key = ("BUY", symbol)
                if key not in last_sent:
                    tg_send(format_signal(symbol, "BUY", buys, per_ex_info[buys[0]]))
                    last_sent.add(key)

            # SELL confirm
            if len(sells) >= MIN_EXCH_CONFIRM:
                key = ("SELL", symbol)
                if key not in last_sent:
                    tg_send(format_signal(symbol, "SELL", sells, per_ex_info[sells[0]]))
                    last_sent.add(key)

        time.sleep(SCAN_EVERY_MINUTES * 60)

if __name__ == "__main__":
    main_loop()
