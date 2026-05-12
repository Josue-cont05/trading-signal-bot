import itertools
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from backtesting.report import calculate_metrics
from config.settings import DAILY_INTERVAL
from services.twelvedata_service import TwelveDataService
from strategies.smc_v1 import STRATEGY_ID, TIMEFRAME
from strategies.smc_v1 import add_daily_indicators, add_entry_indicators
from strategies.smc_v1 import detect_bos, detect_bos_short
from strategies.smc_v1 import find_unmitigated_imbalance, find_unmitigated_imbalance_short
from strategies.smc_v1 import find_yfc, find_yfc_short


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("strategies.smc_v1").setLevel(logging.WARNING)

FILTERS = ["macro", "bos", "imbalance", "yfc"]
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
COMMISSION_PERCENT = 0.002
SLIPPAGE_PERCENT = 0.001
MAX_SIGNALS_PER_MONTH = 3
SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD"]
DAILY_HISTORY_LIMIT = 1500
ENTRY_HISTORY_LIMIT = 3000
MAX_STOP_DISTANCE_PERCENT = 3.0
MIN_RR = 3.0


def run_sensitivity_analysis() -> None:
    service = TwelveDataService()
    combinations = _filter_combinations()
    all_results = []

    for active_filters in combinations:
        logger.info("Testing filters: %s", "+".join(active_filters))
        for symbol in SYMBOLS:
            try:
                result = _run_combination(service, symbol, active_filters)
                all_results.append(result)
                logger.info(
                    "%s %s trades=%s pf=%s wr=%s",
                    symbol,
                    "+".join(active_filters),
                    result["total_trades"],
                    result["profit_factor"],
                    result["win_rate"],
                )
            except Exception as exc:
                logger.exception("Sensitivity run failed for %s %s: %s", symbol, "+".join(active_filters), exc)

    print_sensitivity_report(all_results)


def _filter_combinations() -> list[list[str]]:
    return [list(combo) for size in range(2, 5) for combo in itertools.combinations(FILTERS, size)]


def _run_combination(service: TwelveDataService, symbol: str, active_filters: list[str]) -> dict:
    daily_df = service.get_historical_klines(symbol, DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    entry_df = service.get_historical_klines(symbol, TIMEFRAME, total_limit=ENTRY_HISTORY_LIMIT)
    daily_prepared = add_daily_indicators(daily_df).dropna().reset_index(drop=True)
    entry_prepared = add_entry_indicators(entry_df).dropna().reset_index(drop=True)

    trades = []
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

        signal = evaluate_smc_flexible(symbol, daily_slice, entry_slice, active_filters)
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
        "filters": active_filters,
        "filter_count": len(active_filters),
        "symbol": symbol,
        "total_trades": metrics["total_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "net_pnl": metrics["net_profit_loss"],
        "balance_final": metrics["balance_final"],
        "max_drawdown_percent": metrics["max_drawdown_percent"],
        "trades": trades,
    }


def evaluate_smc_flexible(
    symbol: str,
    daily: pd.DataFrame,
    entry: pd.DataFrame,
    active_filters: list[str],
) -> dict | None:
    if len(daily) < 2 or len(entry) < 60:
        return None

    daily_last = daily.iloc[-1]
    daily_ema_50 = float(daily_last["ema_50"])
    daily_ema_200 = float(daily_last["ema_200"])
    daily_close = float(daily_last["close"])
    price = float(entry.iloc[-1]["close"])
    atr = float(entry.iloc[-1]["atr_14"])

    macro_long = daily_ema_50 > daily_ema_200 and daily_close > daily_ema_50
    macro_short = daily_ema_50 < daily_ema_200 and daily_close < daily_ema_50
    macro_valid = macro_long or macro_short
    bos_detected = detect_bos(entry) or detect_bos_short(entry)
    imbalance_found = (
        find_unmitigated_imbalance(entry, price) is not None
        or find_unmitigated_imbalance_short(entry, price) is not None
    )
    yfc_found = find_yfc(entry) is not None or find_yfc_short(entry) is not None

    if "macro" in active_filters and not macro_valid:
        return None
    if "bos" in active_filters and not bos_detected:
        return None
    if "imbalance" in active_filters and not imbalance_found:
        return None
    if "yfc" in active_filters and not yfc_found:
        return None
    if atr <= 0 or not np.isfinite(atr):
        return None

    directions = []
    if macro_long:
        directions.append("long")
    if macro_short:
        directions.append("short")
    if not directions:
        directions = ["long", "short"]

    for direction in directions:
        signal = _build_flexible_signal(symbol, direction, entry, price, atr, active_filters)
        if signal:
            return signal

    return None


def _build_flexible_signal(
    symbol: str,
    direction: str,
    entry: pd.DataFrame,
    price: float,
    atr: float,
    active_filters: list[str],
) -> dict | None:
    is_long = direction == "long"
    bos_detected = detect_bos(entry) if is_long else detect_bos_short(entry)
    imbalance_level = (
        find_unmitigated_imbalance(entry, price)
        if is_long
        else find_unmitigated_imbalance_short(entry, price)
    )
    yfc_level = find_yfc(entry) if is_long else find_yfc_short(entry)

    if "bos" in active_filters and not bos_detected:
        return None
    if "imbalance" in active_filters and imbalance_level is None:
        return None
    if "yfc" in active_filters and yfc_level is None:
        return None

    if imbalance_level is None:
        imbalance_level = price
    if yfc_level is None:
        yfc_level = float(entry.tail(10)["low"].min()) if is_long else float(entry.tail(10)["high"].max())

    if is_long:
        stop_loss = yfc_level - (atr * 0.25)
        risk = price - stop_loss
        take_profit_1 = price + (risk * 1.5)
        take_profit_2 = price + (risk * 3.0)
        stop_type = "yfc_low"
    else:
        stop_loss = yfc_level + (atr * 0.25)
        risk = stop_loss - price
        take_profit_1 = price - (risk * 1.5)
        take_profit_2 = price - (risk * 3.0)
        stop_type = "yfc_high"

    stop_distance_percent = (risk / price) * 100 if price else 0.0
    if risk <= 0 or stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
        return None

    rr_actual = abs(take_profit_2 - price) / risk
    if rr_actual < MIN_RR:
        return None

    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": TIMEFRAME,
        "direction": direction,
        "entry_price": round(price, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit_1": round(take_profit_1, 5),
        "take_profit_2": round(take_profit_2, 5),
        "risk_reward": "TP1 50% 1:1.5 / TP2 50% 1:3",
        "status": "active",
        "stop_type": stop_type,
        "reasons": [
            f"Filters: {'+'.join(active_filters)}",
            f"Direction: {direction.upper()}",
            f"Imbalance level: {imbalance_level:.5f}",
            f"Stop distance: {stop_distance_percent:.2f}%",
            f"ATR: {atr:.5f}",
        ],
    }


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


def print_sensitivity_report(results: list[dict]) -> None:
    aggregated = _aggregate_by_filters(results)

    print("\nSENSITIVITY ANALYSIS REPORT")
    print("============================")
    print("\nBY FILTER COUNT:")
    for filter_count in [2, 3, 4]:
        candidates = [item for item in aggregated if item["filter_count"] == filter_count and item["total_trades"] > 0]
        if not candidates:
            print(f"{filter_count} filtros - sin operaciones")
            continue
        best = max(candidates, key=lambda item: item["profit_factor"])
        print(
            f"{filter_count} filtros - mejor combinacion: {best['filter_key']} | "
            f"PF: {best['profit_factor']} | Trades: {best['total_trades']} | WR: {best['win_rate']}%"
        )

    top = sorted(
        [item for item in aggregated if item["total_trades"] > 0],
        key=lambda item: item["profit_factor"],
        reverse=True,
    )[:5]
    print("\nTOP 5 COMBINACIONES (por profit factor):")
    for index, item in enumerate(top, start=1):
        print(
            f"{index}. {item['filter_key']} | PF: {item['profit_factor']} | "
            f"Trades: {item['total_trades']} | WR: {item['win_rate']}% | DD: {item['max_drawdown_percent']}%"
        )

    profitable = [item for item in aggregated if item["profit_factor"] > 1.0 and item["total_trades"] > 0]
    counter = Counter(filter_name for item in profitable for filter_name in item["filters"])
    if counter:
        best_filter, count = counter.most_common(1)[0]
        print("\nFILTRO MAS VALIOSO:")
        print(
            f"El filtro que aparece en mas combinaciones rentables (PF > 1.0) es: {best_filter.upper()}"
        )
        print(f"Aparece en {count} de {len(profitable)} combinaciones rentables.")
    else:
        best_filter = "N/A"
        print("\nFILTRO MAS VALIOSO:")
        print("No hay combinaciones rentables con PF > 1.0.")

    best_combination = top[0]["filter_key"] if top else "N/A"
    print("\nRECOMENDACION:")
    print(f"Filtro base: {best_filter.upper() if best_filter != 'N/A' else best_filter}")
    print(f"Mejor combinacion encontrada: {best_combination}")


def _aggregate_by_filters(results: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for result in results:
        filter_key = "+".join(result["filters"])
        grouped.setdefault(
            filter_key,
            {
                "filters": result["filters"],
                "filter_count": result["filter_count"],
                "filter_key": filter_key,
                "trades": [],
            },
        )
        grouped[filter_key]["trades"].extend(result.get("trades", []))

    aggregated = []
    for group in grouped.values():
        metrics = calculate_metrics(group["trades"], INITIAL_CAPITAL)
        aggregated.append(
            {
                "filters": group["filters"],
                "filter_count": group["filter_count"],
                "filter_key": group["filter_key"],
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "net_pnl": metrics["net_profit_loss"],
                "balance_final": metrics["balance_final"],
                "max_drawdown_percent": metrics["max_drawdown_percent"],
            }
        )
    return aggregated


if __name__ == "__main__":
    run_sensitivity_analysis()
