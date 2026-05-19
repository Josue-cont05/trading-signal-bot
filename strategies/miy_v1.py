import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
STRATEGY_ID = "miy_v1"
STRATEGY_NAME = "MIY V1 - Macro + Imbalance + YFC"
TIMEFRAME = "4h"
DAILY_TIMEFRAME = "1d"
MAX_SIGNALS_PER_MONTH = 4
MAX_STOP_DISTANCE_PERCENT = 4.0
MIN_RR = 3.0
VALID_SYMBOLS = ["NAS100"]
VALID_DIRECTIONS = ["long", "short"]


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()
    return data


def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["atr_14"] = calculate_atr(data, period=14)
    return data


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


def evaluate_miy_v1(symbol: str, daily_df: pd.DataFrame, entry_df: pd.DataFrame) -> dict | None:
    if symbol not in VALID_SYMBOLS:
        logger.info("%s skipped because MIY V1 supports only %s.", symbol, ", ".join(VALID_SYMBOLS))
        return None
    if len(daily_df) < 220 or len(entry_df) < 60:
        logger.warning("Not enough candles to evaluate %s miy_v1.", symbol)
        return None

    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()
    if len(daily) < 2 or len(entry) < 60:
        logger.warning("Indicator calculation produced insufficient miy_v1 data for %s.", symbol)
        return None

    return evaluate_miy_v1_prepared(symbol, daily, entry)


def evaluate_miy_v1_prepared(
    symbol: str,
    daily: pd.DataFrame,
    entry: pd.DataFrame,
    ignore_session: bool = False,
) -> dict | None:
    if symbol not in VALID_SYMBOLS or len(daily) < 2 or len(entry) < 60:
        return None

    daily_last = daily.iloc[-1]
    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    daily_close = float(daily_last["close"])
    current_price = float(entry.iloc[-1]["close"])
    session_valid = is_valid_session() if not ignore_session else True
    atr = float(entry.iloc[-1]["atr_14"])

    macro_valid_long = daily_ema_50 > daily_ema_200 and daily_close > daily_ema_50
    long_signal = _build_signal(
        symbol=symbol,
        direction="long",
        macro_valid=macro_valid_long,
        entry=entry,
        current_price=current_price,
        atr=atr,
        session_valid=session_valid,
        daily_ema_50=daily_ema_50,
        daily_ema_200=daily_ema_200,
    )
    if long_signal:
        return long_signal

    macro_valid_short = daily_ema_50 < daily_ema_200 and daily_close < daily_ema_50
    return _build_signal(
        symbol=symbol,
        direction="short",
        macro_valid=macro_valid_short,
        entry=entry,
        current_price=current_price,
        atr=atr,
        session_valid=session_valid,
        daily_ema_50=daily_ema_50,
        daily_ema_200=daily_ema_200,
    )


def _build_signal(
    symbol: str,
    direction: str,
    macro_valid: bool,
    entry: pd.DataFrame,
    current_price: float,
    atr: float,
    session_valid: bool,
    daily_ema_50: float,
    daily_ema_200: float,
) -> dict | None:
    if direction not in VALID_DIRECTIONS:
        return None

    is_long = direction == "long"
    imbalance_level = (
        find_unmitigated_imbalance(entry, current_price)
        if is_long
        else find_unmitigated_imbalance_short(entry, current_price)
    )
    yfc_original = find_yfc(entry) if is_long else find_yfc_short(entry)
    yfc_level = yfc_original

    logger.info(
        "%s miy_v1 %s check. macro_valid=%s imbalance=%s yfc=%s session=%s atr=%.5f",
        symbol,
        direction,
        macro_valid,
        imbalance_level is not None,
        yfc_original is not None,
        session_valid,
        atr,
    )

    if not macro_valid:
        return None
    if imbalance_level is None:
        return None
    if not session_valid:
        return None
    if atr <= 0 or not np.isfinite(atr):
        return None

    entry_price = current_price
    if yfc_level is None:
        yfc_level = float(entry.tail(10)["low"].min()) if is_long else float(entry.tail(10)["high"].max())
        yfc_reason = (
            f"YFC fallback (mínimo reciente): {yfc_level:.5f}"
            if is_long
            else f"YFC fallback (máximo reciente): {yfc_level:.5f}"
        )
    else:
        yfc_reason = f"YFC low: {yfc_level:.5f}" if is_long else f"YFC high: {yfc_level:.5f}"

    if is_long:
        stop_loss = yfc_level - (atr * 0.25)
        risk = entry_price - stop_loss
        take_profit_1 = entry_price + (risk * 1.5)
        take_profit_2 = entry_price + (risk * 3.0)
        stop_type = "yfc_low"
    else:
        stop_loss = yfc_level + (atr * 0.25)
        risk = stop_loss - entry_price
        take_profit_1 = entry_price - (risk * 1.5)
        take_profit_2 = entry_price - (risk * 3.0)
        stop_type = "yfc_high"

    stop_distance_percent = (risk / entry_price) * 100 if entry_price else 0.0

    if stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
        return None
    if risk <= 0:
        return None

    rr_actual = abs(take_profit_2 - entry_price) / risk
    if rr_actual < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": TIMEFRAME,
        "direction": direction,
        "entry_price": round(entry_price, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit_1": round(take_profit_1, 5),
        "take_profit_2": round(take_profit_2, 5),
        "risk_reward": "TP1 50% 1:1.5 / TP2 50% 1:3",
        "status": "active",
        "stop_type": stop_type,
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Dirección: {'LONG' if direction == 'long' else 'SHORT'}",
            f"Filtro macro: EMA50 {'>' if direction == 'long' else '<'} EMA200",
            f"Símbolo: {symbol}",
            f"Imbalance no mitigado en: {imbalance_level:.5f}",
            f"YFC: {'confirmado' if yfc_original is not None else 'fallback'}",
            yfc_reason,
            f"Stop distance: {stop_distance_percent:.2f}%",
            f"ATR: {atr:.5f}",
            f"RR efectivo: 1:{rr_actual:.1f}",
        ],
    }


def detect_bos(entry_df: pd.DataFrame) -> bool:
    """Detect a break above the prior 10-candle higher high inside the last 30 candles."""
    if len(entry_df) < 11:
        return False

    lookback = entry_df.tail(30).reset_index(drop=True)
    current_price = float(lookback.iloc[-1]["close"])
    start_index = max(10, len(lookback) - 20)

    for index in range(start_index, len(lookback)):
        previous_high = float(lookback.iloc[index - 10 : index]["high"].max())
        candle_high = float(lookback.iloc[index]["high"])
        if candle_high > previous_high and current_price > previous_high:
            return True

    return False


def detect_bos_short(entry_df: pd.DataFrame) -> bool:
    """Detect a break below the prior 10-candle lower low inside the last 30 candles."""
    if len(entry_df) < 11:
        return False

    lookback = entry_df.tail(30).reset_index(drop=True)
    current_price = float(lookback.iloc[-1]["close"])
    start_index = max(10, len(lookback) - 20)

    for index in range(start_index, len(lookback)):
        previous_low = float(lookback.iloc[index - 10 : index]["low"].min())
        candle_low = float(lookback.iloc[index]["low"])
        if candle_low < previous_low and current_price < previous_low:
            return True

    return False


def find_unmitigated_imbalance(entry_df: pd.DataFrame, current_price: float) -> float | None:
    """Return the midpoint of the most recent bullish imbalance in the last 50 candles."""
    if len(entry_df) < 3:
        return None

    lookback = entry_df.tail(50).reset_index(drop=True)
    for index in range(len(lookback) - 3, -1, -1):
        first_high = float(lookback.iloc[index]["high"])
        third_low = float(lookback.iloc[index + 2]["low"])
        if first_high < third_low and current_price > first_high:
            return (first_high + third_low) / 2

    return None


def find_unmitigated_imbalance_short(entry_df: pd.DataFrame, current_price: float) -> float | None:
    """Return the midpoint of the most recent bearish imbalance in the last 50 candles."""
    if len(entry_df) < 3:
        return None

    lookback = entry_df.tail(50).reset_index(drop=True)
    for index in range(len(lookback) - 3, -1, -1):
        first_low = float(lookback.iloc[index]["low"])
        third_high = float(lookback.iloc[index + 2]["high"])
        if first_low > third_high and current_price < first_low:
            return (first_low + third_high) / 2

    return None


def find_yfc(entry_df: pd.DataFrame) -> float | None:
    """Find the low of the most recent refined bullish foundational candle."""
    if len(entry_df) < 3:
        return None

    lookback = entry_df.tail(20).reset_index(drop=True)
    for index in range(len(lookback) - 2, 0, -1):
        candle = lookback.iloc[index]
        next_candle = lookback.iloc[index + 1]
        is_bearish_yfc = float(candle["close"]) < float(candle["open"])
        engulfed_by_next = float(next_candle["close"]) > float(candle["open"])
        has_near_imbalance = float(lookback.iloc[index - 1]["high"]) < float(next_candle["low"])

        if is_bearish_yfc and engulfed_by_next and has_near_imbalance:
            return float(candle["low"])

    return None


def find_yfc_short(entry_df: pd.DataFrame) -> float | None:
    """Find the high of the most recent refined bearish foundational candle."""
    if len(entry_df) < 3:
        return None

    lookback = entry_df.tail(20).reset_index(drop=True)
    for index in range(len(lookback) - 2, 0, -1):
        candle = lookback.iloc[index]
        next_candle = lookback.iloc[index + 1]
        is_bullish_yfc = float(candle["close"]) > float(candle["open"])
        engulfed_by_next = float(next_candle["close"]) < float(candle["open"])
        has_near_imbalance = float(lookback.iloc[index - 1]["low"]) > float(next_candle["high"])

        if is_bullish_yfc and engulfed_by_next and has_near_imbalance:
            return float(candle["high"])

    return None


def is_valid_session() -> bool:
    """Allow entries only during London or New York killzone windows in UTC."""
    current_hour = datetime.now(timezone.utc).hour
    london_session = 7 <= current_hour < 10
    new_york_session = 13 <= current_hour < 16
    return london_session or new_york_session
