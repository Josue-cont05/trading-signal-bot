import logging

import numpy as np
import pandas as pd

from config.settings import ENTRY_INTERVAL


logger = logging.getLogger(__name__)
STRATEGY_ID = "swing_long_v2"
STRATEGY_NAME = "Swing LONG V2"
EMA21_PROXIMITY_PERCENT = 2.5
MIN_SCORE_REQUIRED = 4


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    return data


def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
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


def evaluate_swing_long_v2(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if len(daily_df) < 200 or len(entry_df) < 60:
        logger.warning("Not enough candles to evaluate %s swing v2.", symbol)
        return None

    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()
    if daily.empty or len(entry) < 2:
        logger.warning("Indicator calculation produced insufficient swing v2 data for %s.", symbol)
        return None

    daily_last = daily.iloc[-1]
    entry_last = entry.iloc[-1]
    entry_prev = entry.iloc[-2]

    daily_close = float(daily_last["close"])
    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    trend_daily = daily_ema_50 > daily_ema_200 and daily_close > daily_ema_50

    price = float(entry_last["close"])
    previous_close = float(entry_prev["close"])
    ema_21 = float(entry_last["ema_21"])
    ema_distance_percent = abs(price - ema_21) / ema_21 * 100
    price_near_ema21 = ema_distance_percent <= EMA21_PROXIMITY_PERCENT

    rsi_current = float(entry_last["rsi_14"])
    rsi_previous = float(entry_prev["rsi_14"])
    rsi_in_range = 45 <= rsi_current <= 60
    rsi_recovering = rsi_current > rsi_previous

    macd_hist_current = float(entry_last["macd_histogram"])
    macd_hist_previous = float(entry_prev["macd_histogram"])
    macd_histogram_improving = macd_hist_current > macd_hist_previous
    macd_near_or_above_zero = macd_hist_current >= -abs(price * 0.0005)

    volume_current = float(entry_last["volume"])
    volume_average = float(entry_last["volume_avg_20"])
    volume_ratio = volume_current / volume_average if volume_average else 0.0
    volume_above_average = volume_current > volume_average

    bullish_candle = price > previous_close
    atr_current = float(entry_last["atr_14"])

    checks = {
        "precio cerca EMA21": price_near_ema21,
        "RSI en rango": rsi_in_range,
        "RSI recuperándose": rsi_recovering,
        "MACD histogram mejorando": macd_histogram_improving,
        "volumen sobre promedio": volume_above_average,
        "vela alcista": bullish_candle,
    }
    score = sum(1 for passed in checks.values() if passed)
    passed_conditions = [name for name, passed in checks.items() if passed]
    failed_conditions = [name for name, passed in checks.items() if not passed]

    logger.info(
        "%s swing_v2 check. trend_daily=%s score=%s/6 min=%s rsi=%.2f->%.2f "
        "macd_hist=%.6f->%.6f volume=%.2f/%.2f atr=%.4f ema_distance=%.2f%% "
        "macd_near_zero=%s passed=%s failed=%s",
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
        ema_distance_percent,
        macd_near_or_above_zero,
        passed_conditions,
        failed_conditions,
    )

    if not trend_daily:
        return None

    if score < MIN_SCORE_REQUIRED:
        return None

    recent_low_stop = float(entry.tail(10)["low"].min())
    atr_stop = price - (atr_current * 1.5)
    stop_loss = min(recent_low_stop, atr_stop)
    risk = price - stop_loss

    if risk <= 0 or not np.isfinite(risk):
        logger.info("%s swing_v2 skipped because calculated risk is not valid.", symbol)
        return None

    take_profit_1 = price + (risk * 2)
    take_profit_2 = price + (risk * 4)
    confidence = "alta" if score >= 5 else "media"

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": ENTRY_INTERVAL,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "1:2 / 1:4",
        "status": "active",
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Score técnico: {score}/6",
            f"Nivel de confianza: {confidence}",
            f"EMA diaria: EMA50 {daily_ema_50:.4f} > EMA200 {daily_ema_200:.4f}; precio diario {daily_close:.4f} > EMA50 ({trend_daily})",
            f"RSI actual y anterior: {rsi_current:.2f} vs {rsi_previous:.2f}",
            f"MACD histogram actual y anterior: {macd_hist_current:.6f} vs {macd_hist_previous:.6f}",
            f"Volumen actual vs promedio 20: {volume_current:.2f} vs {volume_average:.2f} ({volume_ratio:.2f}x)",
            f"ATR usado: {atr_current:.4f}",
            f"Distancia a EMA21 4H: {ema_distance_percent:.2f}%",
            f"MACD histogram cerca/sobre 0: {macd_near_or_above_zero}",
            f"Stop conservador: min(mínimo 10 velas {recent_low_stop:.4f}, precio - ATR*1.5 {atr_stop:.4f})",
            f"Condiciones cumplidas: {', '.join(passed_conditions)}",
            f"Condiciones fallidas: {', '.join(failed_conditions) if failed_conditions else 'ninguna'}",
        ],
    }
