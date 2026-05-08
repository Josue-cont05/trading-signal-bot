import logging

import numpy as np
import pandas as pd

from config.settings import EMA_PROXIMITY_PERCENT, ENTRY_INTERVAL


logger = logging.getLogger(__name__)
STRATEGY_ID = "swing_long_v1"
STRATEGY_NAME = "Professional Swing LONG V1"
MIN_SCORE_REQUIRED = 3


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    data["volume_avg_20"] = data["volume"].rolling(window=20).mean()
    data["atr_14"] = calculate_atr(data, period=14)
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


def evaluate_swing_long_v1(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if len(daily_df) < 200 or len(entry_df) < 60:
        logger.warning("Not enough candles to evaluate %s professional swing.", symbol)
        return None

    daily = add_indicators(daily_df).dropna()
    entry = add_indicators(entry_df).dropna()

    if daily.empty or len(entry) < 2:
        logger.warning("Indicator calculation produced insufficient data for %s.", symbol)
        return None

    daily_last = daily.iloc[-1]
    entry_last = entry.iloc[-1]
    entry_prev = entry.iloc[-2]

    daily_close = float(daily_last["close"])
    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    daily_trend = daily_ema_50 > daily_ema_200
    daily_price_above_ema50 = daily_close > daily_ema_50

    price = float(entry_last["close"])
    previous_close = float(entry_prev["close"])
    ema_50_4h = float(entry_last["ema_50"])
    ema_distance_percent = abs(price - ema_50_4h) / ema_50_4h * 100
    price_near_ema = ema_distance_percent <= EMA_PROXIMITY_PERCENT

    rsi_current = float(entry_last["rsi_14"])
    rsi_previous = float(entry_prev["rsi_14"])
    rsi_in_range = 40 <= rsi_current <= 70
    rsi_recovering = rsi_current > rsi_previous

    volume_current = float(entry_last["volume"])
    volume_average = float(entry_last["volume_avg_20"])
    volume_ratio = volume_current / volume_average if volume_average else 0.0
    volume_above_average = volume_current > volume_average

    bullish_candle_confirmation = price > previous_close
    atr_current = float(entry_last["atr_14"])

    checks = {
        "price_near_ema": price_near_ema,
        "rsi_in_range": rsi_in_range,
        "rsi_recovering": rsi_recovering,
        "volume_above_average": volume_above_average,
        "bullish_candle_confirmation": bullish_candle_confirmation,
    }
    score = sum(1 for passed in checks.values() if passed)
    passed_conditions = [name for name, passed in checks.items() if passed]
    failed_conditions = [name for name, passed in checks.items() if not passed]

    logger.info(
        "%s swing check. score=%s/5 minimum_required=%s trend=%s daily_price_above_ema50=%s "
        "rsi=%.2f->%.2f ema_distance=%.2f%% volume_ratio=%.2f bullish_candle=%s atr=%.4f "
        "passed=%s failed=%s",
        symbol,
        score,
        MIN_SCORE_REQUIRED,
        daily_trend,
        daily_price_above_ema50,
        rsi_previous,
        rsi_current,
        ema_distance_percent,
        volume_ratio,
        bullish_candle_confirmation,
        atr_current,
        passed_conditions,
        failed_conditions,
    )

    if not daily_trend:
        logger.info(
            "%s swing skipped. Daily trend failed: ema50=%.4f ema200=%.4f daily_close=%.4f",
            symbol,
            daily_ema_50,
            daily_ema_200,
            daily_close,
        )
        return None

    if score < MIN_SCORE_REQUIRED:
        logger.info("%s swing skipped. score=%s/5 minimum_required=%s", symbol, score, MIN_SCORE_REQUIRED)
        return None

    recent_low = float(entry.tail(10)["low"].min())
    stop_loss = recent_low - atr_current
    risk = price - stop_loss

    if risk <= 0 or not np.isfinite(risk):
        logger.info("%s swing skipped because calculated risk is not valid.", symbol)
        return None

    take_profit_1 = price + (risk * 2.5)
    take_profit_2 = price + (risk * 4)
    confidence = "alta" if score >= 4 else "media"

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": ENTRY_INTERVAL,
        "entry_price": round(price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_reward": "1:2.5 / 1:4",
        "status": "active",
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Score técnico: {score}/5",
            f"Nivel de confianza: {confidence}",
            f"Tendencia diaria obligatoria: EMA50 > EMA200 ({daily_trend})",
            f"Confirmación extra: precio diario > EMA50 ({daily_price_above_ema50})",
            f"RSI 4H actual/anterior: {rsi_current:.2f} / {rsi_previous:.2f}",
            f"Distancia a EMA50 4H: {ema_distance_percent:.2f}%",
            f"Volumen actual/promedio 20: {volume_current:.2f} / {volume_average:.2f} ({volume_ratio:.2f}x)",
            f"Confirmación alcista 4H: cierre actual {price:.4f} > cierre anterior {previous_close:.4f} ({bullish_candle_confirmation})",
            f"ATR 14 4H usado como buffer: {atr_current:.4f}",
            f"Stop loss: mínimo 10 velas 4H {recent_low:.4f} - ATR {atr_current:.4f}",
            "Take profits ajustados: TP1 1:2.5 y TP2 1:4",
            f"Condiciones cumplidas: {', '.join(passed_conditions)}",
            f"Condiciones fallidas: {', '.join(failed_conditions) if failed_conditions else 'ninguna'}",
        ],
    }
