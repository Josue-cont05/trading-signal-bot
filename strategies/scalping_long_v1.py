import logging

import numpy as np
import pandas as pd

from config.settings import SCALPING_ENTRY_INTERVAL


logger = logging.getLogger(__name__)
STRATEGY_ID = "scalping_long_v1"
STRATEGY_NAME = "Scalping LONG V1"
MIN_SCORE_REQUIRED = 4


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_21"] = data["close"].ewm(span=21, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["volume_avg_20"] = data["volume"].rolling(window=20).mean()
    data["atr_14"] = calculate_atr(data, period=14)
    data["atr_avg_20"] = data["atr_14"].rolling(window=20).mean()
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


def evaluate_scalping_long_v1(symbol: str, entry_df: pd.DataFrame) -> dict | None:
    if len(entry_df) < 50:
        logger.warning("Not enough 15m candles to evaluate %s scalping.", symbol)
        return None

    entry = add_indicators(entry_df).dropna()
    if len(entry) < 2:
        logger.warning("Indicator calculation produced insufficient scalping data for %s.", symbol)
        return None

    current = entry.iloc[-1]
    previous = entry.iloc[-2]

    price = float(current["close"])
    ema_9 = float(current["ema_9"])
    ema_21 = float(current["ema_21"])
    rsi_current = float(current["rsi_14"])
    rsi_previous = float(previous["rsi_14"])
    volume_current = float(current["volume"])
    volume_average = float(current["volume_avg_20"])
    atr_current = float(current["atr_14"])
    atr_average = float(current["atr_avg_20"])

    checks = {
        "EMA 9 > EMA 21": ema_9 > ema_21,
        "Precio sobre EMA 9": price > ema_9,
        "RSI entre 50 y 70": 50 <= rsi_current <= 70,
        "RSI actual mayor que anterior": rsi_current > rsi_previous,
        "Volumen sobre promedio 20": volume_current > volume_average,
        "ATR sobre promedio 20": atr_current > atr_average,
    }
    score = sum(1 for passed in checks.values() if passed)
    passed_conditions = [name for name, passed in checks.items() if passed]
    failed_conditions = [name for name, passed in checks.items() if not passed]

    if score < MIN_SCORE_REQUIRED:
        logger.info(
            "%s scalping skipped. score=%s/6 minimum_required=%s passed=%s failed=%s",
            symbol,
            score,
            MIN_SCORE_REQUIRED,
            passed_conditions,
            failed_conditions,
        )
        return None

    recent_low_stop = float(entry.tail(5)["low"].min())
    atr_stop = price - atr_current
    stop_loss = min(recent_low_stop, atr_stop)
    risk = price - stop_loss

    if risk <= 0 or not np.isfinite(risk):
        logger.info("%s scalping skipped because calculated risk is not valid.", symbol)
        return None

    take_profit_1 = price + (risk * 1.5)
    take_profit_2 = price + (risk * 2)
    confidence = "alta" if score >= 5 else "media"

    logger.info(
        "%s scalping accepted. score=%s/6 minimum_required=%s passed=%s failed=%s confidence=%s",
        symbol,
        score,
        MIN_SCORE_REQUIRED,
        passed_conditions,
        failed_conditions,
        confidence,
    )

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": SCALPING_ENTRY_INTERVAL,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "1:1.5 / 1:2",
        "status": "active",
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Score técnico: {score}/6",
            f"Nivel de confianza: {confidence}",
            f"EMA 9 > EMA 21: {ema_9:.4f} > {ema_21:.4f} ({checks['EMA 9 > EMA 21']})",
            f"Precio actual por encima de EMA 9: {price:.4f} > {ema_9:.4f} ({checks['Precio sobre EMA 9']})",
            f"RSI actual y anterior: {rsi_current:.2f} vs {rsi_previous:.2f}",
            f"Volumen actual vs promedio 20: {volume_current:.2f} vs {volume_average:.2f}",
            f"ATR actual vs promedio 20: {atr_current:.4f} vs {atr_average:.4f}",
            f"Condiciones cumplidas: {', '.join(passed_conditions)}",
            f"Condiciones fallidas: {', '.join(failed_conditions) if failed_conditions else 'ninguna'}",
            f"Stop loss conservador: min(mínimo 5 velas {recent_low_stop:.4f}, precio - ATR {atr_stop:.4f})",
            "Take profits calculados con relación 1:1.5 y 1:2",
        ],
    }
