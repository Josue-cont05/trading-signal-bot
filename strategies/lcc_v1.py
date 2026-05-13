import logging
from datetime import datetime, timezone
 
import numpy as np
import pandas as pd
 
 
logger = logging.getLogger(__name__)
 
STRATEGY_ID = "lcc_v1"
STRATEGY_NAME = "LCC V1 - Liquidez, Cobertura y Confirmación Multitemporal"
TIMEFRAME = "1h"
MACRO_TIMEFRAME = "4h"
MAX_SIGNALS_PER_MONTH = 4
MAX_STOP_DISTANCE_PERCENT = 2.0
MIN_RR = 3.0
VALID_SYMBOLS = ["EURUSD", "GBPUSD", "XAUUSD"]
VALID_DIRECTIONS = ["long", "short"]
 
# Parámetros de la lógica LCC
H4_LOOKBACK = 30           # velas H4 para evaluar contexto macro
H4_PREMIUM_PCT = 0.60      # por encima del 60% del rango → zona prima (SHORT)
H4_DISCOUNT_PCT = 0.40     # por debajo del 40% del rango → zona descuento (LONG)
H1_SWING_LOOKBACK = 10     # velas H1 para identificar swing relevante
H1_COVER_BODY_PCT = 0.20   # cobertura mínima del cuerpo de la vela sweep
M15_LOOKBACK = 8           # velas M15 para detectar desplazamiento/CHoCH
M15_MIN_BODY_RATIO = 0.40  # vela de desplazamiento debe tener cuerpo >= 40% del rango
MIN_SCORE = 7              # score mínimo sobre 10 para emitir señal
 
 
# ---------------------------------------------------------------------------
# Puntuación LCC
# Sweep H1:       +2
# Cobertura H1:   +3
# Alineación H4:  +3
# Confirmación M15: +2 (M1 omitido en backtest — solo para ejecución manual)
# Total máximo:   10
# ---------------------------------------------------------------------------
 
def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """H4 no necesita indicadores clásicos — usa OHLC puro."""
    return df.copy()
 
 
def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ATR14 al dataframe H1."""
    data = df.copy()
    data["atr_14"] = _calculate_atr(data, period=14)
    return data
 
 
def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
 
 
# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------
 
def evaluate_lcc_v1(
    symbol: str,
    daily_df: pd.DataFrame,
    entry_df: pd.DataFrame,
) -> dict | None:
    """Punto de entrada para el scanner en producción."""
    if symbol not in VALID_SYMBOLS:
        logger.info("%s skipped — LCC V1 soporta solo %s.", symbol, ", ".join(VALID_SYMBOLS))
        return None
    if len(daily_df) < H4_LOOKBACK or len(entry_df) < H1_SWING_LOOKBACK + 5:
        logger.warning("Datos insuficientes para evaluar %s lcc_v1.", symbol)
        return None
 
    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()
 
    if len(daily) < H4_LOOKBACK or len(entry) < H1_SWING_LOOKBACK + 5:
        return None
 
    return evaluate_lcc_v1_prepared(symbol, daily, entry)
 
 
def evaluate_lcc_v1_prepared(
    symbol: str,
    daily: pd.DataFrame,
    entry: pd.DataFrame,
    ignore_session: bool = False,
) -> dict | None:
    """Versión para backtesting."""
    if symbol not in VALID_SYMBOLS:
        return None
    if len(daily) < H4_LOOKBACK or len(entry) < H1_SWING_LOOKBACK + 5:
        return None
 
    session_valid = is_valid_session() if not ignore_session else True
    if not session_valid:
        return None
 
    current_price = float(entry.iloc[-1]["close"])
    atr = float(entry.iloc[-1]["atr_14"]) if "atr_14" in entry.columns else _fallback_atr(entry)
 
    if atr <= 0 or not np.isfinite(atr):
        return None
 
    # H4 determina dirección candidata
    h4_context, h4_score, h4_reasons = _analyze_h4_context(daily)
    if h4_context == "neutral":
        logger.info("%s lcc_v1 — contexto H4 neutral, no operar.", symbol)
        return None
 
    direction = "long" if h4_context == "discount" else "short"
 
    return _build_signal(
        symbol=symbol,
        direction=direction,
        entry=entry,
        current_price=current_price,
        atr=atr,
        h4_score=h4_score,
        h4_reasons=h4_reasons,
    )
 
 
# ---------------------------------------------------------------------------
# H4 — Contexto macro prima/descuento
# ---------------------------------------------------------------------------
 
def _analyze_h4_context(
    h4: pd.DataFrame,
) -> tuple[str, int, list[str]]:
    """
    Evalúa si el precio está en zona de prima o descuento institucional.
    Retorna (contexto, score_H4, reasons).
    Score H4: +3 si zona clara, +0 si neutral.
    """
    reasons: list[str] = []
    if len(h4) < H4_LOOKBACK:
        return "neutral", 0, ["H4: datos insuficientes"]
 
    window = h4.tail(H4_LOOKBACK)
    high = float(window["high"].max())
    low = float(window["low"].min())
    rng = high - low
 
    if rng == 0:
        return "neutral", 0, ["H4: rango cero"]
 
    current_price = float(h4.iloc[-1]["close"])
    position = (current_price - low) / rng
 
    if position <= H4_DISCOUNT_PCT:
        reasons.append(
            f"H4: zona descuento institucional ({position:.1%} del rango H4) → LONG candidato ✓"
        )
        return "discount", 3, reasons
 
    if position >= H4_PREMIUM_PCT:
        reasons.append(
            f"H4: zona prima institucional ({position:.1%} del rango H4) → SHORT candidato ✓"
        )
        return "premium", 3, reasons
 
    reasons.append(f"H4: zona neutral ({position:.1%} del rango H4) — no operar")
    return "neutral", 0, reasons
 
 
# ---------------------------------------------------------------------------
# H1 — Barrido de liquidez (Sweep) + Cobertura
# ---------------------------------------------------------------------------
 
def _detect_sweep_and_cover(
    h1: pd.DataFrame,
    direction: str,
) -> tuple[bool, bool, float, list[str]]:
    """
    Detecta la secuencia:
      1. Vela sweep: barre el swing relevante y cierra recuperándolo (cuerpo)
      2. Vela cobertura: siguiente vela cubre al menos H1_COVER_BODY_PCT del cuerpo del sweep
 
    Retorna (sweep_ok, cover_ok, nivel_de_liquidez, reasons).
    """
    reasons: list[str] = []
    if len(h1) < H1_SWING_LOOKBACK + 3:
        return False, False, 0.0, ["H1: datos insuficientes"]
 
    # Las últimas 2 velas cerradas son sweep_candle e cover_candle
    sweep_candle = h1.iloc[-2]
    cover_candle = h1.iloc[-1]
    history = h1.iloc[-(H1_SWING_LOOKBACK + 2):-2]
 
    if direction == "long":
        # Swing low relevante
        swing_level = float(history["low"].min())
        sweep_low = float(sweep_candle["low"])
        sweep_close = float(sweep_candle["close"])
        sweep_open = float(sweep_candle["open"])
 
        # Sweep: barre mínimo y cierra recuperándolo
        swept = sweep_low < swing_level
        recovered = sweep_close > swing_level
 
        if not swept:
            reasons.append(f"H1 LONG: precio no barrió el mínimo relevante {swing_level:.5f}")
            return False, False, swing_level, reasons
 
        if not recovered:
            reasons.append(
                f"H1 LONG: barrido de {swing_level:.5f} pero sin recuperación del cuerpo"
            )
            return True, False, swing_level, reasons
 
        reasons.append(f"H1 LONG: sweep del mínimo {swing_level:.5f} con recuperación ✓ (+2)")
 
        # Cobertura: siguiente vela cubre el cuerpo del sweep
        sweep_body = abs(sweep_close - sweep_open)
        cover_body = abs(float(cover_candle["close"]) - float(cover_candle["open"]))
        cover_up = float(cover_candle["close"]) > sweep_close
 
        if sweep_body > 0:
            cover_ratio = cover_body / sweep_body
        else:
            cover_ratio = 1.0
 
        cover_ok = cover_up and cover_ratio >= H1_COVER_BODY_PCT
 
        if cover_ok:
            reasons.append(
                f"H1 LONG: cobertura institucional validada "
                f"({cover_ratio:.0%} del cuerpo sweep) ✓ (+3)"
            )
        else:
            reasons.append(
                f"H1 LONG: cobertura insuficiente "
                f"({cover_ratio:.0%} < {H1_COVER_BODY_PCT:.0%} requerido)"
            )
 
        return True, cover_ok, swing_level, reasons
 
    else:  # short
        # Swing high relevante
        swing_level = float(history["high"].max())
        sweep_high = float(sweep_candle["high"])
        sweep_close = float(sweep_candle["close"])
        sweep_open = float(sweep_candle["open"])
 
        # Sweep: barre máximo y cierra recuperándolo (con cuerpo por debajo)
        swept = sweep_high > swing_level
        recovered = sweep_close < swing_level
 
        if not swept:
            reasons.append(f"H1 SHORT: precio no barrió el máximo relevante {swing_level:.5f}")
            return False, False, swing_level, reasons
 
        if not recovered:
            reasons.append(
                f"H1 SHORT: barrido de {swing_level:.5f} pero sin recuperación del cuerpo"
            )
            return True, False, swing_level, reasons
 
        reasons.append(f"H1 SHORT: sweep del máximo {swing_level:.5f} con recuperación ✓ (+2)")
 
        # Cobertura
        sweep_body = abs(sweep_close - sweep_open)
        cover_body = abs(float(cover_candle["close"]) - float(cover_candle["open"]))
        cover_down = float(cover_candle["close"]) < sweep_close
 
        if sweep_body > 0:
            cover_ratio = cover_body / sweep_body
        else:
            cover_ratio = 1.0
 
        cover_ok = cover_down and cover_ratio >= H1_COVER_BODY_PCT
 
        if cover_ok:
            reasons.append(
                f"H1 SHORT: cobertura institucional validada "
                f"({cover_ratio:.0%} del cuerpo sweep) ✓ (+3)"
            )
        else:
            reasons.append(
                f"H1 SHORT: cobertura insuficiente "
                f"({cover_ratio:.0%} < {H1_COVER_BODY_PCT:.0%} requerido)"
            )
 
        return True, cover_ok, swing_level, reasons
 
 
# ---------------------------------------------------------------------------
# M15 — Confirmación fractal (desplazamiento estructural)
# ---------------------------------------------------------------------------
 
def _confirm_m15_displacement(
    h1: pd.DataFrame,
    direction: str,
) -> tuple[bool, list[str]]:
    """
    Resamplea H1 a M15 y busca un desplazamiento estructural (CHoCH):
    - Una vela M15 con cuerpo >= M15_MIN_BODY_RATIO del rango de la vela
    - Que rompa el máximo/mínimo de las velas M15 anteriores
    """
    reasons: list[str] = []
 
    # Resample H1 → M15
    try:
        if not isinstance(h1.index, pd.DatetimeIndex):
            # Intentar usar open_time como índice
            if "open_time" in h1.columns:
                h1_indexed = h1.set_index("open_time")
            else:
                reasons.append("M15: no se puede resamplear sin índice datetime")
                return False, reasons
        else:
            h1_indexed = h1
 
        m15 = h1_indexed.resample("15min").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna()
    except Exception as exc:
        reasons.append(f"M15: resample falló ({exc}), usando H1 directamente")
        m15 = h1.copy()
 
    if len(m15) < M15_LOOKBACK + 2:
        reasons.append("M15: datos insuficientes para confirmación")
        return False, reasons
 
    window = m15.tail(M15_LOOKBACK + 1)
    last = window.iloc[-1]
    prev = window.iloc[:-1]
 
    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_range = last_high - last_low
 
    if last_range == 0:
        reasons.append("M15: vela de rango cero")
        return False, reasons
 
    body = abs(last_close - last_open)
    body_ratio = body / last_range
 
    if direction == "long":
        broke_structure = last_close > float(prev["high"].max())
        is_bullish = last_close > last_open
        displacement = broke_structure and is_bullish and body_ratio >= M15_MIN_BODY_RATIO
 
        if displacement:
            reasons.append(
                f"M15: desplazamiento alcista confirmado "
                f"(cuerpo={body_ratio:.0%}, rompió máximos previos) ✓ (+2)"
            )
            return True, reasons
        else:
            reasons.append(
                f"M15: desplazamiento insuficiente "
                f"(broke={broke_structure}, alcista={is_bullish}, cuerpo={body_ratio:.0%})"
            )
            return False, reasons
 
    else:  # short
        broke_structure = last_close < float(prev["low"].min())
        is_bearish = last_close < last_open
        displacement = broke_structure and is_bearish and body_ratio >= M15_MIN_BODY_RATIO
 
        if displacement:
            reasons.append(
                f"M15: desplazamiento bajista confirmado "
                f"(cuerpo={body_ratio:.0%}, rompió mínimos previos) ✓ (+2)"
            )
            return True, reasons
        else:
            reasons.append(
                f"M15: desplazamiento insuficiente "
                f"(broke={broke_structure}, bajista={is_bearish}, cuerpo={body_ratio:.0%})"
            )
            return False, reasons
 
 
# ---------------------------------------------------------------------------
# Constructor de señal
# ---------------------------------------------------------------------------
 
def _build_signal(
    symbol: str,
    direction: str,
    entry: pd.DataFrame,
    current_price: float,
    atr: float,
    h4_score: int,
    h4_reasons: list[str],
) -> dict | None:
    if direction not in VALID_DIRECTIONS:
        return None
 
    is_long = direction == "long"
 
    # H1 — Sweep + Cobertura
    sweep_ok, cover_ok, liquidity_level, h1_reasons = _detect_sweep_and_cover(entry, direction)
 
    # M15 — solo si H1 validó la cobertura (regla crítica)
    if cover_ok:
        m15_ok, m15_reasons = _confirm_m15_displacement(entry, direction)
    else:
        m15_ok = False
        m15_reasons = ["M15: saltado — cobertura H1 no validada (regla crítica)"]
 
    # Score
    score = h4_score  # +3 si H4 ok
    score += 2 if sweep_ok else 0
    score += 3 if cover_ok else 0
    score += 2 if m15_ok else 0
 
    logger.info(
        "%s lcc_v1 %s — sweep=%s cover=%s m15=%s score=%d/10",
        symbol, direction, sweep_ok, cover_ok, m15_ok, score,
    )
 
    if score < MIN_SCORE:
        return None
 
    if liquidity_level == 0.0:
        return None
 
    # Gestión de riesgo — SL detrás del nivel de liquidez
    if is_long:
        stop_loss = liquidity_level - (atr * 0.25)
        risk = current_price - stop_loss
        take_profit_1 = current_price + (risk * 1.5)
        take_profit_2 = current_price + (risk * 3.0)
        stop_type = "liquidity_low_h1"
    else:
        stop_loss = liquidity_level + (atr * 0.25)
        risk = stop_loss - current_price
        take_profit_1 = current_price - (risk * 1.5)
        take_profit_2 = current_price - (risk * 3.0)
        stop_type = "liquidity_high_h1"
 
    if risk <= 0:
        return None
 
    stop_distance_percent = (risk / current_price) * 100 if current_price else 0.0
    if stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
        return None
 
    rr_actual = abs(take_profit_2 - current_price) / risk
    if rr_actual < MIN_RR:
        return None
 
    return {
        "symbol": symbol,
        "strategy": STRATEGY_ID,
        "timeframe": TIMEFRAME,
        "direction": direction,
        "entry_price": round(current_price, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit_1": round(take_profit_1, 5),
        "take_profit_2": round(take_profit_2, 5),
        "risk_reward": "TP1 50% 1:1.5 / TP2 50% 1:3",
        "status": "active",
        "stop_type": stop_type,
        "reasons": [
            f"Estrategia: {STRATEGY_NAME}",
            f"Dirección: {'LONG' if is_long else 'SHORT'}",
            f"Símbolo: {symbol}",
            f"Score LCC: {score}/10",
        ] + h4_reasons + h1_reasons + m15_reasons + [
            f"Nivel de liquidez H1: {liquidity_level:.5f}",
            f"Stop distance: {stop_distance_percent:.2f}%",
            f"ATR H1: {atr:.5f}",
            f"RR efectivo: 1:{rr_actual:.1f}",
        ],
    }
 
 
# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
 
def _fallback_atr(df: pd.DataFrame, period: int = 14) -> float:
    series = _calculate_atr(df, period=period)
    val = series.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0
 
 
def is_valid_session() -> bool:
    """Permite entradas en London killzone y solapamiento London-NY (UTC)."""
    current_hour = datetime.now(timezone.utc).hour
    london = 7 <= current_hour < 10
    new_york = 13 <= current_hour < 16
    return london or new_york