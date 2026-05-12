import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
STRATEGY_ID = "swing_long_v9"
STRATEGY_NAME = "Swing LONG V9 Diario + 4H"
TIMEFRAME = "4h"
MACRO_TIMEFRAME = "1d"
MAX_SIGNALS_PER_MONTH = 2
EMA_PROXIMITY_PERCENT = 2.5
MIN_SCORE_REQUIRED = 5
MAX_PRICE_ABOVE_EMA21_PERCENT = 4.0
MAX_ATR_PERCENT = 5.0
MAX_STOP_DISTANCE_PERCENT = 3.0


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["macd_line"], data["macd_signal"], data["macd_histogram"] = calculate_macd(data["close"])
    return data


def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["atr_14"] = calculate_atr(data, period=14)
    data["volume_avg_20"] = data["volume"].rolling(window=20).mean()
    data["macd_line"], data["macd_signal"], data["macd_histogram"] = calculate_macd(data["close"])
    return data


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + relative_strength))).fillna(50)


def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def find_order_block_stop(entry: pd.DataFrame, atr: float, max_stop_percent: float, price: float) -> float | None:
    """Find a valid bullish order block stop inside the latest 50 candles."""
    if len(entry) < 3 or atr <= 0 or price <= 0:
        return None

    lookback = entry.tail(50).reset_index(drop=True)
    for index in range(len(lookback) - 3, -1, -1):
        order_block = lookback.iloc[index]
        if float(order_block["close"]) >= float(order_block["open"]):
            continue

        next_candles = lookback.iloc[index + 1 : index + 3]
        if len(next_candles) < 2:
            continue

        bullish_impulse = all(float(candle["close"]) > float(candle["open"]) for _, candle in next_candles.iterrows())
        if not bullish_impulse:
            continue

        order_block_low = float(order_block["low"])
        stop_loss = order_block_low - (atr * 0.25)
        stop_distance_percent = ((price - stop_loss) / price) * 100
        if stop_loss < price and stop_distance_percent <= max_stop_percent:
            return stop_loss

    return None


def evaluate_swing_long_v9(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if len(daily_df) < 220 or len(entry_df) < 80:
        logger.warning("Not enough candles to evaluate %s swing v9.", symbol)
        return None

    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()
    if len(daily) < 2 or len(entry) < 2:
        logger.warning("Indicator calculation produced insufficient swing v9 data for %s.", symbol)
        return None

    return evaluate_swing_long_v9_prepared(symbol, daily, entry)


def evaluate_swing_long_v9_prepared(symbol: str, daily: pd.DataFrame, entry: pd.DataFrame) -> dict | None:
    if len(daily) < 2 or len(entry) < 2:
        return None

    daily_last = daily.iloc[-1]
    daily_prev = daily.iloc[-2]
    entry_last = entry.iloc[-1]
    entry_prev = entry.iloc[-2]

    daily_close = float(daily_last["close"])
    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    daily_rsi = float(daily_last["rsi_14"])
    daily_macd_hist = float(daily_last["macd_histogram"])
    daily_macd_prev = float(daily_prev["macd_histogram"])
    daily_macd_ok = daily_macd_hist >= 0 or daily_macd_hist > daily_macd_prev
    macro_valid = (
        daily_ema_50 > daily_ema_200
        and daily_close > daily_ema_50
        and daily_rsi > 50
        and daily_macd_ok
    )

    price = float(entry_last["close"])
    ema_21 = float(entry_last["ema_21"])
    ema_50 = float(entry_last["ema_50"])
    ema21_distance_percent = abs(price - ema_21) / ema_21 * 100
    ema50_distance_percent = abs(price - ema_50) / ema_50 * 100
    price_near_ema = min(ema21_distance_percent, ema50_distance_percent) <= EMA_PROXIMITY_PERCENT
    price_above_ema21_percent = ((price - ema_21) / ema_21) * 100

    rsi_current = float(entry_last["rsi_14"])
    rsi_previous = float(entry_prev["rsi_14"])
    rsi_in_range = 45 <= rsi_current <= 62
    rsi_recovering = rsi_current > rsi_previous

    macd_hist_current = float(entry_last["macd_histogram"])
    macd_hist_previous = float(entry_prev["macd_histogram"])
    macd_improving = macd_hist_current > macd_hist_previous

    breaks_previous_high = price > float(entry_prev["high"])
    volume_current = float(entry_last["volume"])
    volume_average = float(entry_last["volume_avg_20"])
    volume_ratio = volume_current / volume_average if volume_average else 0.0
    volume_confirms = volume_current >= volume_average

    atr_current = float(entry_last["atr_14"])
    atr_percent = (atr_current / price) * 100 if price else 0.0
    stop_loss = find_order_block_stop(entry, atr_current, MAX_STOP_DISTANCE_PERCENT, price)
    if stop_loss is None:
        return None
    risk = price - stop_loss
    stop_distance_percent = (risk / price) * 100 if price else 0.0

    checks = {
        "precio cerca EMA21 o EMA50": price_near_ema,
        "RSI en rango": rsi_in_range,
        "RSI recuperándose": rsi_recovering,
        "MACD histogram mejorando": macd_improving,
        "cierre rompe máximo previo": breaks_previous_high,
        "volumen confirma": volume_confirms,
    }
    score = sum(1 for passed in checks.values() if passed)
    passed_conditions = [name for name, passed in checks.items() if passed]
    failed_conditions = [name for name, passed in checks.items() if not passed]

    logger.info(
        "%s swing_v9 check. macro_valid=%s score=%s/6 rsi4h=%.2f->%.2f "
        "macd4h=%.6f->%.6f volume=%.2f/%.2f atr_percent=%.2f "
        "stop_distance_percent=%.2f price_above_ema21=%.2f",
        symbol,
        macro_valid,
        score,
        rsi_previous,
        rsi_current,
        macd_hist_previous,
        macd_hist_current,
        volume_current,
        volume_average,
        atr_percent,
        stop_distance_percent,
        price_above_ema21_percent,
    )

    if not macro_valid:
        return None
    if score < MIN_SCORE_REQUIRED:
        return None
    if risk <= 0 or not np.isfinite(risk):
        return None
    if stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
        return None
    if atr_percent > MAX_ATR_PERCENT:
        return None
    if price_above_ema21_percent > MAX_PRICE_ABOVE_EMA21_PERCENT:
        return None

    take_profit_1 = price + risk
    take_profit_2 = price + (risk * 3)
    confidence = "alta" if score == 6 else "media"

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": TIMEFRAME,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "stop_type": "order_block",
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "TP1 50% 1:1 / TP2 50% 1:3",
        "status": "active",
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Filtro macro diario válido: {macro_valid} (EMA50 {daily_ema_50:.4f} > EMA200 {daily_ema_200:.4f}; cierre {daily_close:.4f} > EMA50)",
            f"Score 4H {score}/6",
            f"Nivel de confianza: {confidence}",
            f"RSI diario: {daily_rsi:.2f}",
            f"MACD diario: {daily_macd_hist:.6f} vs {daily_macd_prev:.6f}",
            f"RSI 4H actual/anterior: {rsi_current:.2f} vs {rsi_previous:.2f}",
            f"MACD 4H actual/anterior: {macd_hist_current:.6f} vs {macd_hist_previous:.6f}",
            f"Volumen actual/promedio: {volume_current:.2f} vs {volume_average:.2f} ({volume_ratio:.2f}x)",
            f"ATR percent: {atr_percent:.2f}%",
            f"Stop distance percent: {stop_distance_percent:.2f}%",
            "Salida parcial TP1/TP2: 50% en TP1 1R, 50% busca TP2 3R, SL restante a break even tras TP1",
            f"Condiciones cumplidas: {', '.join(passed_conditions)}",
            f"Condiciones fallidas: {', '.join(failed_conditions) if failed_conditions else 'ninguna'}",
        ],
    }
