import logging

import numpy as np
import pandas as pd

from config.settings import ENTRY_INTERVAL


logger = logging.getLogger(__name__)
STRATEGY_ID = "swing_long_v3"
STRATEGY_NAME = "Swing LONG V3"
EMA_PROXIMITY_PERCENT = 3.0
MIN_SCORE_REQUIRED = 5
MAX_PRICE_ABOVE_EMA21_PERCENT = 4.0
MAX_ATR_PERCENT = 6.0
MAX_STOP_DISTANCE_PERCENT = 4.0


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    return data


def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["volume_avg_20"] = data["volume"].rolling(window=20).mean()
    data["atr_14"] = calculate_atr(data, period=14)
    data["macd_line"], data["macd_signal"], data["macd_histogram"] = calculate_macd(data["close"])
    return data


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50)


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


def evaluate_swing_long_v3(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if len(daily_df) < 200 or len(entry_df) < 70:
        logger.warning("Not enough candles to evaluate %s swing v3.", symbol)
        return None

    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()
    if daily.empty or len(entry) < 2:
        logger.warning("Indicator calculation produced insufficient swing v3 data for %s.", symbol)
        return None

    daily_last = daily.iloc[-1]
    entry_last = entry.iloc[-1]
    entry_prev = entry.iloc[-2]

    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    trend_daily = daily_ema_50 > daily_ema_200

    price = float(entry_last["close"])
    ema_21 = float(entry_last["ema_21"])
    ema_50 = float(entry_last["ema_50"])
    ema21_distance_percent = abs(price - ema_21) / ema_21 * 100
    ema50_distance_percent = abs(price - ema_50) / ema_50 * 100
    closest_ema_distance_percent = min(ema21_distance_percent, ema50_distance_percent)
    price_near_ema = closest_ema_distance_percent <= EMA_PROXIMITY_PERCENT
    price_above_ema21_percent = ((price - ema_21) / ema_21) * 100

    rsi_current = float(entry_last["rsi_14"])
    rsi_previous = float(entry_prev["rsi_14"])
    rsi_in_range = 45 <= rsi_current <= 65
    rsi_recovering = rsi_current > rsi_previous

    macd_hist_current = float(entry_last["macd_histogram"])
    macd_hist_previous = float(entry_prev["macd_histogram"])
    macd_histogram_improving = macd_hist_current > macd_hist_previous
    macd_positive_or_crossing = macd_hist_current >= 0 or (macd_hist_previous < 0 and macd_hist_current > macd_hist_previous)

    previous_high = float(entry_prev["high"])
    close_breaks_previous_high = price > previous_high

    volume_current = float(entry_last["volume"])
    volume_average = float(entry_last["volume_avg_20"])
    volume_ratio = volume_current / volume_average if volume_average else 0.0
    volume_above_average = volume_current >= volume_average

    atr_current = float(entry_last["atr_14"])
    atr_percent = (atr_current / price) * 100 if price else 0.0

    checks = {
        "precio cerca EMA21 o EMA50": price_near_ema,
        "RSI en rango": rsi_in_range,
        "RSI recuperandose": rsi_recovering,
        "MACD histogram mejorando": macd_histogram_improving,
        "MACD histogram positivo o cruzando": macd_positive_or_crossing,
        "cierre rompe maximo previo": close_breaks_previous_high,
        "volumen sobre promedio": volume_above_average,
    }
    score = sum(1 for passed in checks.values() if passed)
    passed_conditions = [name for name, passed in checks.items() if passed]
    failed_conditions = [name for name, passed in checks.items() if not passed]

    recent_low = float(entry.tail(10)["low"].min())
    stop_loss = recent_low - (atr_current * 0.5)
    risk = price - stop_loss
    stop_distance_percent = (risk / price) * 100 if price else 0.0

    logger.info(
        "%s swing_v3 check. trend_daily=%s score=%s/7 min=%s rsi=%.2f->%.2f "
        "macd_hist=%.6f->%.6f volume=%.2f/%.2f atr=%.4f atr_pct=%.2f "
        "ema21_distance=%.2f%% ema50_distance=%.2f%% price_above_ema21=%.2f%% "
        "stop_distance=%.2f%% previous_high_break=%s passed=%s failed=%s",
        symbol,
        trend_daily,
        score,
        MIN_SCORE_REQUIRED,
        rsi_previous,
        rsi_current,
        macd_hist_previous,
        macd_hist_current,
        volume_current,
        volume_average,
        atr_current,
        atr_percent,
        ema21_distance_percent,
        ema50_distance_percent,
        price_above_ema21_percent,
        stop_distance_percent,
        close_breaks_previous_high,
        passed_conditions,
        failed_conditions,
    )

    if not trend_daily:
        return None
    if price_above_ema21_percent > MAX_PRICE_ABOVE_EMA21_PERCENT:
        logger.info("%s swing_v3 skipped because price is %.2f%% above EMA21.", symbol, price_above_ema21_percent)
        return None
    if atr_percent > MAX_ATR_PERCENT:
        logger.info("%s swing_v3 skipped because ATR percent is %.2f%%.", symbol, atr_percent)
        return None
    if score < MIN_SCORE_REQUIRED:
        return None
    if risk <= 0 or not np.isfinite(risk):
        logger.info("%s swing_v3 skipped because calculated risk is not valid.", symbol)
        return None
    if stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
        logger.info("%s swing_v3 skipped because stop distance is %.2f%%.", symbol, stop_distance_percent)
        return None

    take_profit_1 = price + (risk * 2)
    take_profit_2 = price + (risk * 3)
    confidence = "alta" if score >= 6 else "media"

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": ENTRY_INTERVAL,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "1:2 / 1:3",
        "status": "active",
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Score técnico: {score}/7",
            f"Nivel de confianza: {confidence}",
            f"EMA diaria: EMA50 {daily_ema_50:.4f} > EMA200 {daily_ema_200:.4f}",
            f"RSI actual/anterior: {rsi_current:.2f} vs {rsi_previous:.2f}",
            f"MACD actual/anterior: {macd_hist_current:.6f} vs {macd_hist_previous:.6f}",
            f"Volumen actual/promedio: {volume_current:.2f} vs {volume_average:.2f} ({volume_ratio:.2f}x)",
            f"Distancia EMA: EMA21 {ema21_distance_percent:.2f}%, EMA50 {ema50_distance_percent:.2f}%",
            f"ATR percent: {atr_percent:.2f}%",
            f"Stop distance percent: {stop_distance_percent:.2f}%",
            f"Stop loss: minimo 10 velas {recent_low:.4f} - ATR*0.5 ({atr_current * 0.5:.4f})",
            f"Condiciones cumplidas: {', '.join(passed_conditions)}",
            f"Condiciones fallidas: {', '.join(failed_conditions) if failed_conditions else 'ninguna'}",
        ],
    }
