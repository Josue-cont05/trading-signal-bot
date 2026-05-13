import itertools
import logging
import time
from collections import defaultdict

import strategies.lcc_v1 as lcc_module
import pandas as pd

from backtesting.report import calculate_metrics
from services.twelvedata_service import TwelveDataService
from strategies.lcc_v1 import (
    STRATEGY_ID as LCC_STRATEGY_ID,
    TIMEFRAME as LCC_TIMEFRAME,
    add_daily_indicators as lcc_add_daily,
    add_entry_indicators as lcc_add_entry,
    evaluate_lcc_v1_prepared,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("strategies.lcc_v1").setLevel(logging.WARNING)

PARAM_GRID: dict[str, list] = {
    "MIN_SCORE": [4, 5, 6],
    "H1_COVER_BODY_PCT": [0.10, 0.15, 0.20],
    "H4_PREMIUM_PCT": [0.55, 0.60, 0.65],
    "H1_SWING_LOOKBACK": [5, 10, 15],
}
SYMBOLS: list[str] = ["EURUSD", "GBPUSD", "USDCAD", "XAUUSD"]
LCC_DAILY_INTERVAL = "4h"
LCC_ENTRY_INTERVAL = "1h"
DAILY_HISTORY_LIMIT = 1500
ENTRY_HISTORY_LIMIT = 5000
MAX_SIGNALS_PER_MONTH = 4
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
COMMISSION_PERCENT = 0.002
SLIPPAGE_PERCENT = 0.001
MAX_STOP_DISTANCE_BY_SYMBOL: dict[str, float] = {
    "EURUSD": 2.0,
    "GBPUSD": 2.0,
    "USDCAD": 2.0,
    "XAUUSD": 4.0,
}
_PARAM_ABBREV: dict[str, str] = {
    "MIN_SCORE": "MS",
    "H1_COVER_BODY_PCT": "COV",
    "H4_PREMIUM_PCT": "PREM",
    "H1_SWING_LOOKBACK": "SWING",
}


def run_lcc_sensitivity() -> None:
    service = TwelveDataService()
    combinations = _param_combinations()
    logger.info(
        "LCC V1 sensitivity: %d combinations × %d symbols = %d runs",
        len(combinations),
        len(SYMBOLS),
        len(combinations) * len(SYMBOLS),
    )

    logger.info("Downloading data for all symbols...")
    symbol_data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in SYMBOLS:
        try:
            daily_df = service.get_historical_klines(symbol, LCC_DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
            entry_df = service.get_historical_klines(symbol, LCC_ENTRY_INTERVAL, total_limit=ENTRY_HISTORY_LIMIT)
            daily_prepared = lcc_add_daily(daily_df).dropna().reset_index(drop=True)
            entry_prepared = lcc_add_entry(entry_df).dropna().reset_index(drop=True)
            symbol_data[symbol] = {"daily": daily_prepared, "entry": entry_prepared}
            logger.info("Data ready for %s", symbol)
        except Exception as exc:
            logger.error("Failed to download data for %s: %s", symbol, exc)
        time.sleep(15)

    per_symbol_results: list[dict] = []
    total = len(combinations)
    for idx, params in enumerate(combinations, start=1):
        combo_key = _combo_key(params)
        logger.info("[%d/%d] Testing %s", idx, total, combo_key)
        for symbol in SYMBOLS:
            if symbol not in symbol_data:
                continue
            try:
                result = _run_combination_symbol(
                    symbol,
                    symbol_data[symbol]["daily"],
                    symbol_data[symbol]["entry"],
                    params,
                )
                per_symbol_results.append(result)
                logger.info(
                    "  %s trades=%s pf=%s wr=%s%%",
                    symbol,
                    result["total_trades"],
                    result["profit_factor"],
                    result["win_rate"],
                )
            except Exception as exc:
                logger.exception("Run failed for %s %s: %s", symbol, combo_key, exc)

    print_lcc_sensitivity_report(per_symbol_results)


def _param_combinations() -> list[dict]:
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _combo_key(params: dict) -> str:
    return "|".join(f"{_PARAM_ABBREV.get(k, k)}={v}" for k, v in params.items())


def _run_combination_symbol(
    symbol: str,
    daily_prepared: pd.DataFrame,
    entry_prepared: pd.DataFrame,
    params: dict,
) -> dict:
    orig = {
        "MIN_SCORE": lcc_module.MIN_SCORE,
        "H1_COVER_BODY_PCT": lcc_module.H1_COVER_BODY_PCT,
        "H4_PREMIUM_PCT": lcc_module.H4_PREMIUM_PCT,
        "H4_DISCOUNT_PCT": lcc_module.H4_DISCOUNT_PCT,
        "H1_SWING_LOOKBACK": lcc_module.H1_SWING_LOOKBACK,
        "MAX_STOP_DISTANCE_PERCENT": lcc_module.MAX_STOP_DISTANCE_PERCENT,
        "VALID_SYMBOLS": lcc_module.VALID_SYMBOLS,
    }
    try:
        lcc_module.MIN_SCORE = params["MIN_SCORE"]
        lcc_module.H1_COVER_BODY_PCT = params["H1_COVER_BODY_PCT"]
        lcc_module.H4_PREMIUM_PCT = params["H4_PREMIUM_PCT"]
        lcc_module.H4_DISCOUNT_PCT = 1.0 - params["H4_PREMIUM_PCT"]
        lcc_module.H1_SWING_LOOKBACK = params["H1_SWING_LOOKBACK"]
        lcc_module.MAX_STOP_DISTANCE_PERCENT = 4.0 if symbol == "XAUUSD" else 2.0
        lcc_module.VALID_SYMBOLS = ["EURUSD", "GBPUSD", "USDCAD", "XAUUSD"]

        trades: list[dict] = []
        index = 30
        signals_by_month: dict[str, int] = {}

        while index < len(entry_prepared) - 2:
            entry_slice = entry_prepared.iloc[: index + 1].copy()
            current_close_time = entry_slice.iloc[-1]["close_time"]
            daily_slice = daily_prepared[daily_prepared["close_time"] <= current_close_time].copy()
            if daily_slice.empty:
                index += 1
                continue

            signal_month = entry_slice.iloc[-1]["close_time"].strftime("%Y-%m")
            if signals_by_month.get(signal_month, 0) >= MAX_SIGNALS_PER_MONTH:
                index += 1
                continue

            signal = evaluate_lcc_v1_prepared(symbol, daily_slice, entry_slice, ignore_session=True)
            if not signal:
                index += 1
                continue

            future_df = entry_prepared.iloc[index + 1 :].copy()
            trade = _simulate_partial_exit_trade(signal, future_df, COMMISSION_PERCENT, SLIPPAGE_PERCENT)
            trades.append(trade)
            signals_by_month[signal_month] = signals_by_month.get(signal_month, 0) + 1
            index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_prepared)

        metrics = calculate_metrics(trades, INITIAL_CAPITAL)
        return {
            "combo_key": _combo_key(params),
            "params": params,
            "symbol": symbol,
            "total_trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "net_pnl": metrics["net_profit_loss"],
            "balance_final": metrics["balance_final"],
            "max_drawdown_percent": metrics["max_drawdown_percent"],
            "trades": trades,
        }
    finally:
        for attr, value in orig.items():
            setattr(lcc_module, attr, value)


def _simulate_partial_exit_trade(
    signal: dict,
    future_df: pd.DataFrame,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    entry = float(signal["entry_price"])
    stop = float(signal["stop_loss"])
    tp1 = float(signal["take_profit_1"])
    tp2 = float(signal["take_profit_2"])
    direction = signal.get("direction", "long")
    is_long = direction == "long"
    risk_distance = entry - stop if is_long else stop - entry
    risk_amount = INITIAL_CAPITAL * RISK_PER_TRADE
    position_size = risk_amount / risk_distance if risk_distance > 0 else 0
    half_position = position_size * 0.5

    result = "open"
    exit_price = None
    exit_time = None
    exit_index = None
    tp1_hit = False
    tp1_time = None

    gross_pnl = 0.0
    exit_notional = 0.0

    for candle_index, candle in future_df.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        candle_close_time = candle["close_time"].isoformat()
        exit_time = candle_close_time

        if not tp1_hit:
            stop_hit = low <= stop if is_long else high >= stop
            tp1_reached = high >= tp1 if is_long else low <= tp1
            tp2_reached = high >= tp2 if is_long else low <= tp2

            if stop_hit:
                result = "loss"
                exit_price = stop
                exit_index = int(candle_index)
                gross_pnl = ((stop - entry) if is_long else (entry - stop)) * position_size
                exit_notional = stop * position_size
                break
            if tp1_reached:
                tp1_hit = True
                tp1_time = candle_close_time
                gross_pnl += ((tp1 - entry) if is_long else (entry - tp1)) * half_position
                exit_notional += tp1 * half_position

                if tp2_reached:
                    result = "win_full"
                    exit_price = tp2
                    exit_index = int(candle_index)
                    gross_pnl += ((tp2 - entry) if is_long else (entry - tp2)) * half_position
                    exit_notional += tp2 * half_position
                    break
                continue

        if tp1_hit:
            break_even_hit = low <= entry if is_long else high >= entry
            tp2_reached = high >= tp2 if is_long else low <= tp2

            if break_even_hit:
                result = "partial_win"
                exit_price = entry
                exit_index = int(candle_index)
                exit_notional += entry * half_position
                break
            if tp2_reached:
                result = "win_full"
                exit_price = tp2
                exit_index = int(candle_index)
                gross_pnl += ((tp2 - entry) if is_long else (entry - tp2)) * half_position
                exit_notional += tp2 * half_position
                break

    if result == "open":
        exit_price = float(future_df.iloc[-1]["close"]) if not future_df.empty else entry
        exit_time = future_df.iloc[-1]["close_time"].isoformat() if not future_df.empty else None
        if tp1_hit:
            gross_pnl += ((exit_price - entry) if is_long else (entry - exit_price)) * half_position
            exit_notional += exit_price * half_position
        else:
            gross_pnl = ((exit_price - entry) if is_long else (entry - exit_price)) * position_size if position_size else 0.0
            exit_notional = exit_price * position_size if exit_price is not None else 0.0

    entry_notional = entry * position_size
    commission_cost = (entry_notional + exit_notional) * (commission_percent / 100)
    slippage_cost = (entry_notional + exit_notional) * (slippage_percent / 100)
    net_pnl = gross_pnl - commission_cost - slippage_cost

    return {
        "symbol": signal["symbol"],
        "strategy": signal["strategy"],
        "timeframe": signal["timeframe"],
        "direction": direction,
        "entry_time": future_df.iloc[0]["open_time"].isoformat() if not future_df.empty else None,
        "entry_price": round(entry, 8),
        "stop_loss": round(stop, 8),
        "break_even_stop": round(entry, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "partial_exit_percent": 50,
        "runner_exit_percent": 50,
        "tp1_hit": tp1_hit,
        "tp1_time": tp1_time,
        "result": result,
        "exit_price": round(exit_price, 8) if exit_price is not None else None,
        "exit_time": exit_time,
        "exit_index": exit_index,
        "risk_amount": round(risk_amount, 2),
        "position_size": round(position_size, 8),
        "commission_percent": commission_percent,
        "slippage_percent": slippage_percent,
        "commission_cost": round(commission_cost, 2),
        "slippage_cost": round(slippage_cost, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "pnl": round(net_pnl, 2),
    }


def print_lcc_sensitivity_report(results: list[dict]) -> None:
    aggregated = _aggregate_by_params(results)

    print("\nLCC V1 SENSITIVITY ANALYSIS REPORT")
    print("=====================================")

    top5 = sorted(
        [item for item in aggregated if item["total_trades"] > 0],
        key=lambda item: item["profit_factor"],
        reverse=True,
    )[:5]

    print("\nTOP 5 COMBINACIONES (por profit factor agregado):")
    for rank, item in enumerate(top5, start=1):
        print(
            f"{rank}. {item['combo_key']} | PF: {item['profit_factor']} | "
            f"Trades: {item['total_trades']} | WR: {item['win_rate']}% | DD: {item['max_drawdown_percent']}%"
        )
        print(f"   Mejor par: {item['best_symbol']} (PF: {item['best_symbol_pf']})")

    profitable = [item for item in aggregated if item["profit_factor"] > 1.0 and item["total_trades"] > 0]
    print("\nPARAMETRO MAS VALIOSO:")
    if profitable:
        param_counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for item in profitable:
            for k, v in item["params"].items():
                param_counts[k][v] += 1
        most_valuable = max(
            param_counts.keys(),
            key=lambda k: max(param_counts[k].values()),
        )
        best_value = max(param_counts[most_valuable], key=lambda v: param_counts[most_valuable][v])
        count = param_counts[most_valuable][best_value]
        print(
            f"El valor dominante en combinaciones rentables (PF > 1.0) es: "
            f"{most_valuable}={best_value}"
        )
        print(f"Aparece en {count} de {len(profitable)} combinaciones rentables.")
    else:
        print("No hay combinaciones rentables con PF > 1.0.")

    print("\nRECOMENDACION DE PARAMETROS OPTIMOS:")
    if top5:
        best = top5[0]
        for k, v in best["params"].items():
            print(f"  {k}: {v}")
        print(
            f"Basado en: mejor profit factor agregado ({best['profit_factor']}) "
            f"con {best['total_trades']} trades totales."
        )
    else:
        print("  Sin combinaciones con operaciones suficientes.")


def _aggregate_by_params(results: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for result in results:
        combo_key = result["combo_key"]
        grouped.setdefault(
            combo_key,
            {
                "combo_key": combo_key,
                "params": result["params"],
                "trades": [],
                "symbol_results": [],
            },
        )
        grouped[combo_key]["trades"].extend(result.get("trades", []))
        grouped[combo_key]["symbol_results"].append(result)

    aggregated = []
    for group in grouped.values():
        metrics = calculate_metrics(group["trades"], INITIAL_CAPITAL)
        symbol_results_with_trades = [r for r in group["symbol_results"] if r["total_trades"] > 0]
        if symbol_results_with_trades:
            best_sym = max(symbol_results_with_trades, key=lambda r: r["profit_factor"])
            best_symbol = best_sym["symbol"]
            best_symbol_pf = best_sym["profit_factor"]
        else:
            best_symbol = "N/A"
            best_symbol_pf = 0.0
        aggregated.append(
            {
                "combo_key": group["combo_key"],
                "params": group["params"],
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "net_pnl": metrics["net_profit_loss"],
                "balance_final": metrics["balance_final"],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
                "best_symbol": best_symbol,
                "best_symbol_pf": best_symbol_pf,
            }
        )
    return aggregated


if __name__ == "__main__":
    run_lcc_sensitivity()
