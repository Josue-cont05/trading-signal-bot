import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.report import calculate_metrics
from config.settings import BASE_DIR
from services.binance_service import BinanceService


logger = logging.getLogger(__name__)

RESULTS_PATH = BASE_DIR / "database" / "dynamic_backtest_results.json"
ALLOWED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
ALLOWED_TIMEFRAMES = {"15m", "1h", "4h", "1d"}
ALLOWED_LOOKBACK_DAYS = {90, 180, 365}
INTERVAL_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


DEFAULT_CONFIG = {
    "symbol": "BTCUSDT",
    "timeframe": "4h",
    "lookback_days": 180,
    "initial_balance": 1000.0,
    "risk_per_trade_percent": 1.0,
    "commission_percent": 0.1,
    "slippage_percent": 0.05,
    "use_ema_filter": True,
    "ema_fast_period": 21,
    "ema_slow_period": 50,
    "ema_condition": "fast_above_slow",
    "use_rsi_filter": True,
    "rsi_period": 14,
    "rsi_min": 45,
    "rsi_max": 65,
    "require_rsi_recovering": True,
    "use_volume_filter": True,
    "volume_avg_period": 20,
    "volume_multiplier": 1.0,
    "use_macd_filter": True,
    "require_macd_histogram_improving": True,
    "use_atr_filter": True,
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
    "minimum_score": 4,
    "take_profit_rr": 2.0,
    "stop_loss_mode": "recent_low_or_atr",
    "recent_low_period": 10,
}


def run_dynamic_backtest(raw_config: dict) -> dict:
    config = _normalize_config(raw_config)
    logger.info("Starting dynamic backtest with config=%s", config)

    service = BinanceService()
    candles = _download_candles(service, config["symbol"], config["timeframe"], config["lookback_days"])
    data = _add_indicators(candles, config).dropna().reset_index(drop=True)

    if len(data) < _minimum_warmup(config) + 5:
        raise ValueError("Not enough historical candles after indicator warmup.")

    trades = _run_simulation(data, config)
    metrics = calculate_metrics(trades, config["initial_balance"])
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "trades": trades,
    }
    save_dynamic_result(result)
    logger.info("Dynamic backtest finished. trades=%s net=%s", metrics["total_trades"], metrics["net_profit_loss"])
    return result


def get_last_dynamic_result() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def save_dynamic_result(result: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_config(raw_config: dict) -> dict:
    config = DEFAULT_CONFIG | (raw_config or {})
    config["symbol"] = str(config["symbol"]).upper()
    config["timeframe"] = str(config["timeframe"])
    config["lookback_days"] = int(config["lookback_days"])

    if config["symbol"] not in ALLOWED_SYMBOLS:
        raise ValueError("Invalid symbol.")
    if config["timeframe"] not in ALLOWED_TIMEFRAMES:
        raise ValueError("Invalid timeframe.")
    if config["lookback_days"] not in ALLOWED_LOOKBACK_DAYS:
        raise ValueError("lookback_days must be 90, 180, or 365.")

    numeric_fields = [
        "initial_balance",
        "risk_per_trade_percent",
        "commission_percent",
        "slippage_percent",
        "volume_multiplier",
        "atr_stop_multiplier",
        "take_profit_rr",
    ]
    int_fields = [
        "ema_fast_period",
        "ema_slow_period",
        "rsi_period",
        "volume_avg_period",
        "atr_period",
        "minimum_score",
        "recent_low_period",
    ]
    bool_fields = [
        "use_ema_filter",
        "use_rsi_filter",
        "require_rsi_recovering",
        "use_volume_filter",
        "use_macd_filter",
        "require_macd_histogram_improving",
        "use_atr_filter",
    ]

    for field in numeric_fields:
        config[field] = float(config[field])
    for field in int_fields:
        config[field] = int(config[field])
    for field in bool_fields:
        config[field] = bool(config[field])

    if config["initial_balance"] <= 0:
        raise ValueError("initial_balance must be positive.")
    if not 0 < config["risk_per_trade_percent"] <= 10:
        raise ValueError("risk_per_trade_percent must be between 0 and 10.")
    if config["take_profit_rr"] <= 0:
        raise ValueError("take_profit_rr must be positive.")
    if config["stop_loss_mode"] not in {"recent_low", "atr", "recent_low_or_atr"}:
        raise ValueError("Invalid stop_loss_mode.")
    if config["ema_condition"] != "fast_above_slow":
        raise ValueError("Only ema_condition=fast_above_slow is supported.")

    return config


def _download_candles(service: BinanceService, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    candles_per_day = 1440 / INTERVAL_MINUTES[timeframe]
    total_limit = min(int(candles_per_day * lookback_days) + 250, 36000)
    return service.get_historical_klines(symbol, timeframe, total_limit=total_limit)


def _add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = df.copy()
    data["ema_fast"] = data["close"].ewm(span=config["ema_fast_period"], adjust=False).mean()
    data["ema_slow"] = data["close"].ewm(span=config["ema_slow_period"], adjust=False).mean()
    data["rsi"] = _calculate_rsi(data["close"], config["rsi_period"])
    data["volume_avg"] = data["volume"].rolling(window=config["volume_avg_period"]).mean()
    data["atr"] = _calculate_atr(data, config["atr_period"])
    data["atr_avg"] = data["atr"].rolling(window=20).mean()
    data["macd_line"], data["macd_signal"], data["macd_histogram"] = _calculate_macd(data["close"])
    return data


def _run_simulation(data: pd.DataFrame, config: dict) -> list[dict]:
    trades = []
    index = _minimum_warmup(config)
    balance = config["initial_balance"]

    while index < len(data) - 2:
        current = data.iloc[index]
        previous = data.iloc[index - 1]
        score, conditions_met = _evaluate_conditions(current, previous, config)

        if score < config["minimum_score"]:
            index += 1
            continue

        stop_loss = _calculate_stop_loss(data.iloc[: index + 1], current, config)
        entry_price = float(current["close"])
        risk_distance = entry_price - stop_loss
        if risk_distance <= 0 or not np.isfinite(risk_distance):
            index += 1
            continue

        take_profit = entry_price + (risk_distance * config["take_profit_rr"])
        future = data.iloc[index + 1 :].copy()
        trade = _simulate_trade(config, current, future, stop_loss, take_profit, score, conditions_met, balance)
        balance += trade["net_pnl"] if trade["result"] in {"win", "loss"} else 0.0
        trade["balance_after_trade"] = round(balance, 2)
        trades.append(trade)
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(data)

    return trades


def _evaluate_conditions(current: pd.Series, previous: pd.Series, config: dict) -> tuple[int, list[str]]:
    score = 0
    met = []

    if config["use_ema_filter"] and float(current["ema_fast"]) > float(current["ema_slow"]):
        score += 1
        met.append("EMA fast above slow")

    if config["use_rsi_filter"] and config["rsi_min"] <= float(current["rsi"]) <= config["rsi_max"]:
        score += 1
        met.append("RSI in range")

    if config["require_rsi_recovering"] and float(current["rsi"]) > float(previous["rsi"]):
        score += 1
        met.append("RSI recovering")

    required_volume = float(current["volume_avg"]) * config["volume_multiplier"]
    if config["use_volume_filter"] and float(current["volume"]) > required_volume:
        score += 1
        met.append("Volume above threshold")

    if config["use_macd_filter"]:
        macd_ok = True
        if config["require_macd_histogram_improving"]:
            macd_ok = float(current["macd_histogram"]) > float(previous["macd_histogram"])
        if macd_ok:
            score += 1
            met.append("MACD histogram improving")

    if config["use_atr_filter"] and float(current["atr"]) > float(current["atr_avg"]):
        score += 1
        met.append("ATR above average")

    return score, met


def _calculate_stop_loss(history: pd.DataFrame, current: pd.Series, config: dict) -> float:
    recent_low = float(history.tail(config["recent_low_period"])["low"].min())
    atr_stop = float(current["close"]) - (float(current["atr"]) * config["atr_stop_multiplier"])

    if config["stop_loss_mode"] == "recent_low":
        return recent_low
    if config["stop_loss_mode"] == "atr":
        return atr_stop
    return min(recent_low, atr_stop)


def _simulate_trade(
    config: dict,
    signal_candle: pd.Series,
    future: pd.DataFrame,
    stop_loss: float,
    take_profit: float,
    score: int,
    conditions_met: list[str],
    balance: float,
) -> dict:
    entry_price = float(signal_candle["close"])
    risk_amount = balance * (config["risk_per_trade_percent"] / 100)
    risk_distance = entry_price - stop_loss
    position_size = risk_amount / risk_distance if risk_distance > 0 else 0

    result = "open"
    exit_price = float(future.iloc[-1]["close"]) if not future.empty else entry_price
    exit_time = future.iloc[-1]["close_time"].isoformat() if not future.empty else None
    exit_index = None

    for candle_index, candle in future.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        if low <= stop_loss:
            result = "loss"
            exit_price = stop_loss
            exit_time = candle["close_time"].isoformat()
            exit_index = int(candle_index)
            break
        if high >= take_profit:
            result = "win"
            exit_price = take_profit
            exit_time = candle["close_time"].isoformat()
            exit_index = int(candle_index)
            break

    gross_pnl = (exit_price - entry_price) * position_size if position_size else 0.0
    entry_notional = entry_price * position_size
    exit_notional = exit_price * position_size
    commission_cost = (entry_notional + exit_notional) * (config["commission_percent"] / 100)
    slippage_cost = (entry_notional + exit_notional) * (config["slippage_percent"] / 100)
    net_pnl = gross_pnl - commission_cost - slippage_cost

    return {
        "entry_time": signal_candle["close_time"].isoformat(),
        "exit_time": exit_time,
        "exit_index": exit_index,
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "entry_price": round(entry_price, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit": round(take_profit, 8),
        "result": result,
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "pnl": round(net_pnl, 2),
        "commission_cost": round(commission_cost, 2),
        "slippage_cost": round(slippage_cost, 2),
        "balance_after_trade": round(balance, 2),
        "score": score,
        "conditions_met": conditions_met,
    }


def _minimum_warmup(config: dict) -> int:
    return max(
        config["ema_fast_period"],
        config["ema_slow_period"],
        config["rsi_period"],
        config["volume_avg_period"],
        config["atr_period"],
        config["recent_low_period"],
        35,
    ) + 5


def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + relative_strength))).fillna(50)


def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
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


def _calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
