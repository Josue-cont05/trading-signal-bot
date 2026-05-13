import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
STRATEGY_ID = "lcc_v1"
STRATEGY_NAME = "LCC V1 - Liquidez, Cobertura y Confirmación"
TIMEFRAME = "1h"
DAILY_TIMEFRAME = "4h"
MAX_STOP_DISTANCE_PERCENT = 4.0
MIN_RR = 3.0
VALID_SYMBOLS = ["XAUUSD"]
VALID_DIRECTIONS = ["long", "short"]

# Parámetros de la lógica LCC
H4_LOOKBACK = 20          # velas H4 para calcular rango macro prima/descuento
H4_PREMIUM_PCT = 0.60     # por encima del 60 % del rango → zona prima
H4_DISCOUNT_PCT = 0.40    # por debajo del 40 % del rango → zona descuento
H1_SWING_LOOKBACK = 10    # velas H1 para identificar swing relevante
H1_COVER_BODY_PCT = 0.30  # la vela de cobertura debe cubrir ≥ 30 % del cuerpo del sweep
M15_DISP_LOOKBACK = 6     # velas M15 para detectar desplazamiento / CHoCH
M15_MIN_DISP_FACTOR = 0.15  # desplazamiento mínimo como factor del ATR M15
MIN_SCORE = 8             # score mínimo sobre 10 para emitir señal
MAX_SIGNALS_PER_MONTH = 4


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara H4 (daily_df en la convención del proyecto).
    LCC V1 no usa indicadores clásicos en H4 — solo OHLC para análisis de rango."""
    return df.copy()


def add_entry_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ATR al dataframe H1 (entry_df)."""
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


# ---------------------------------------------------------------------------
# Funciones principales (convención del proyecto)
# ---------------------------------------------------------------------------

def evaluate_lcc_v1(
    symbol: str,
    daily_df: pd.DataFrame,
    entry_df: pd.DataFrame,
) -> dict | None:
    """
    Punto de entrada para el scanner en producción.
    daily_df → H4 (contexto macro prima/descuento)
    entry_df → H1 (liquidez + cobertura; M15 se deriva por resample)
    """
    if symbol not in VALID_SYMBOLS:
        logger.info("%s skipped — LCC V1 soporta solo %s.", symbol, ", ".join(VALID_SYMBOLS))
        return None
    if len(daily_df) < H4_LOOKBACK or len(entry_df) < H1_SWING_LOOKBACK + 3:
        logger.warning("Datos insuficientes para evaluar %s lcc_v1.", symbol)
        return None

    daily = add_daily_indicators(daily_df).dropna()
    entry = add_entry_indicators(entry_df).dropna()

    if len(daily) < H4_LOOKBACK or len(entry) < H1_SWING_LOOKBACK + 3:
        logger.warning("Indicadores insuficientes para %s lcc_v1.", symbol)
        return None

    return evaluate_lcc_v1_prepared(symbol, daily, entry)


def evaluate_lcc_v1_prepared(
    symbol: str,
    daily: pd.DataFrame,
    entry: pd.DataFrame,
    ignore_session: bool = False,
) -> dict | None:
    """
    Versión para backtesting.
    Evalúa la dirección que indica el contexto H4 y retorna la señal
    si supera el score mínimo, o None si no califica.
    """
    if symbol not in VALID_SYMBOLS or len(daily) < H4_LOOKBACK or len(entry) < H1_SWING_LOOKBACK + 3:
        return None

    current_price = float(entry.iloc[-1]["close"])
    atr = (
        float(entry.iloc[-1]["atr_14"])
        if "atr_14" in entry.columns
        else _fallback_atr(entry)
    )
    session_valid = is_valid_session() if not ignore_session else True

    if not session_valid:
        return None
    if atr <= 0 or not np.isfinite(atr):
        return None

    # H4 — determina la única dirección candidata
    h4_context, h4_reasons = _analyze_h4_context(daily)
    if h4_context == "neutral":
        logger.info("%s lcc_v1 — contexto H4 ambiguo, no operar.", symbol)
        return None

    candidate_direction = "long" if h4_context == "discount" else "short"

    return _build_signal(
        symbol=symbol,
        direction=candidate_direction,
        entry=entry,
        current_price=current_price,
        atr=atr,
        h4_context=h4_context,
        h4_reasons=h4_reasons,
    )
def _analyze_h4_context(
    h4: pd.DataFrame,
) -> tuple[str, list[str]]:
    """
    Determina si el precio actual está en zona de prima o descuento
    basándose en el rango de las últimas H4_LOOKBACK velas.
    Prima  → precio > 60% del rango → candidato SHORT
    Descuento → precio < 40% del rango → candidato LONG
    Neutral → zona media, no operar
    """
    reasons: list[str] = []
    if len(h4) < H4_LOOKBACK:
        return "neutral", ["H4: datos insuficientes para análisis de rango"]

    window = h4.tail(H4_LOOKBACK)
    high = float(window["high"].max())
    low = float(window["low"].min())
    rng = high - low

    if rng == 0:
        return "neutral", ["H4: rango cero, no operar"]

    current_price = float(h4.iloc[-1]["close"])
    position = (current_price - low) / rng

    if position >= H4_PREMIUM_PCT:
        reasons.append(f"H4: zona prima ({position:.1%} del rango) → SHORT candidato")
        return "premium", reasons
    elif position <= H4_DISCOUNT_PCT:
        reasons.append(f"H4: zona descuento ({position:.1%} del rango) → LONG candidato")
        return "discount", reasons
    else:
        reasons.append(f"H4: zona neutral ({position:.1%} del rango) → no operar")
        return "neutral", reasons

# ---------------------------------------------------------------------------
# Constructor de señal
# ---------------------------------------------------------------------------

def _build_signal(
    symbol: str,
    direction: str,
    entry: pd.DataFrame,
    current_price: float,
    atr: float,
    h4_context: str,
    h4_reasons: list[str],
) -> dict | None:
    if direction not in VALID_DIRECTIONS:
        return None

    is_long = direction == "long"

    # Liquidez + Cobertura H1 (regla crítica — no bajar antes de esto)
    h1_valid, sweep_level, h1_reasons = _detect_liquidity_and_cover(entry, direction)

    # Confirmación M15 (solo si H1 validó)
    m15 = _resample_to_m15(entry)
    if h1_valid:
        m15_valid, m15_reasons = _confirm_displacement(m15, direction)
    else:
        m15_valid, m15_reasons = False, ["M15: saltado (cobertura H1 no validada)"]

    # Score
    score = _compute_score(h4_valid=True, h1_valid=h1_valid, m15_valid=m15_valid)

    logger.info(
        "%s lcc_v1 %s — h4=%s liquidez+cobertura=%s confirmacion=%s score=%d/10",
        symbol, direction, h4_context, h1_valid, m15_valid, score,
    )

    if score < MIN_SCORE:
        return None

    # Gestión de riesgo — SL detrás del extremo barrido en H1
    if is_long:
        stop_loss = sweep_level - (atr * 0.25)
        risk = current_price - stop_loss
        take_profit_1 = current_price + (risk * 1.5)
        take_profit_2 = current_price + (risk * 3.0)
        stop_type = "liquidity_low_h1"
    else:
        stop_loss = sweep_level + (atr * 0.25)
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
            f"Contexto H4: {'descuento' if is_long else 'prima'} institucional",
            f"Símbolo: {symbol}",
        ] + h4_reasons + h1_reasons + m15_reasons + [
            f"Score LCC: {score}/10",
            f"Nivel de liquidez H1: {sweep_level:.5f}",
            f"Stop distance: {stop_distance_percent:.2f}%",
            f"ATR H1: {atr:.5f}",
            f"RR efectivo: 1:{rr_actual:.1f}",
        ],
    }


# ---------------------------------------------------------------------------
# L — Liquidez + C — Cobertura (H1)
# ---------------------------------------------------------------------------

def _detect_liquidity_and_cover(
    h1: pd.DataFrame,
    direction: str,
) -> tuple[bool, float, list[str]]:
    """
    L → barrido del swing relevante con recuperación del cuerpo en la misma vela.
    C → la vela siguiente cubre la vela de manipulación.
    Vela [-2] = vela de liquidez (sweep), vela [-1] = vela de cobertura.
    Regla crítica: no confirmar M15 hasta que esta secuencia esté completa.
    Retorna (válido, nivel_de_liquidez, reasons).
    """
    reasons: list[str] = []
    if len(h1) < H1_SWING_LOOKBACK + 3:
        return False, 0.0, ["H1: datos insuficientes para detección de liquidez"]

    sweep_candle = h1.iloc[-2]
    cover_candle = h1.iloc[-1]

    if direction == "long":
        swing_low = _find_swing_low(h1.iloc[:-2], H1_SWING_LOOKBACK)
        if swing_low is None:
            return False, 0.0, ["H1 LONG: sin mínimo relevante identificado"]

        # L: el precio barre el mínimo y cierra recuperándolo
        swept = float(sweep_candle["low"]) < swing_low
        recovered = float(sweep_candle["close"]) > swing_low

        if not (swept and recovered):
            reasons.append(f"H1 LONG: sin barrido de liquidez del mínimo {swing_low:.5f}")
            return False, swing_low, reasons

        reasons.append(f"H1 LONG: liquidez barrida en {swing_low:.5f} con recuperación ✓")

        # C: la siguiente vela cubre la vela de manipulación
        sweep_body = abs(float(sweep_candle["close"]) - float(sweep_candle["open"]))
        cover_body = abs(float(cover_candle["close"]) - float(cover_candle["open"]))
        cover_ok = (
            float(cover_candle["close"]) > float(sweep_candle["open"])
            and (sweep_body == 0 or cover_body >= sweep_body * H1_COVER_BODY_PCT)
        )

        if not cover_ok:
            reasons.append("H1 LONG: cobertura insuficiente — esperar siguiente vela H1")
            return False, swing_low, reasons

        reasons.append("H1 LONG: cobertura institucional validada ✓")
        return True, swing_low, reasons

    else:  # short
        swing_high = _find_swing_high(h1.iloc[:-2], H1_SWING_LOOKBACK)
        if swing_high is None:
            return False, 0.0, ["H1 SHORT: sin máximo relevante identificado"]

        # L: el precio barre el máximo y cierra recuperándolo
        swept = float(sweep_candle["high"]) > swing_high
        recovered = float(sweep_candle["close"]) < swing_high

        if not (swept and recovered):
            reasons.append(f"H1 SHORT: sin barrido de liquidez del máximo {swing_high:.5f}")
            return False, swing_high, reasons

        reasons.append(f"H1 SHORT: liquidez barrida en {swing_high:.5f} con recuperación ✓")

        # C: la siguiente vela cubre la vela de manipulación
        sweep_body = abs(float(sweep_candle["close"]) - float(sweep_candle["open"]))
        cover_body = abs(float(cover_candle["close"]) - float(cover_candle["open"]))
        cover_ok = (
            float(cover_candle["close"]) < float(sweep_candle["open"])
            and (sweep_body == 0 or cover_body >= sweep_body * H1_COVER_BODY_PCT)
        )

        if not cover_ok:
            reasons.append("H1 SHORT: cobertura insuficiente — esperar siguiente vela H1")
            return False, swing_high, reasons

        reasons.append("H1 SHORT: cobertura institucional validada ✓")
        return True, swing_high, reasons


# ---------------------------------------------------------------------------
# C — Confirmación (M15)
# ---------------------------------------------------------------------------

def _confirm_displacement(
    m15: pd.DataFrame,
    direction: str,
) -> tuple[bool, list[str]]:
    """
    Confirma CHoCH / desplazamiento estructural en M15.
    Solo se ejecuta después de que L+C en H1 están validados.
    """
    reasons: list[str] = []
    if len(m15) < M15_DISP_LOOKBACK + 2:
        return False, ["M15: datos insuficientes para confirmación"]

    atr_m15 = float(calculate_atr(m15, period=min(14, len(m15) - 1)).iloc[-1])
    if not np.isfinite(atr_m15) or atr_m15 == 0:
        return False, ["M15: ATR no calculable"]

    window = m15.tail(M15_DISP_LOOKBACK + 1)
    min_move = atr_m15 * M15_MIN_DISP_FACTOR
    last_close = float(window.iloc[-1]["close"])

    if direction == "long":
        prev_highs_max = float(window["high"].iloc[:-1].max())
        displacement = last_close > prev_highs_max
        move = last_close - float(window["low"].min())
    else:
        prev_lows_min = float(window["low"].iloc[:-1].min())
        displacement = last_close < prev_lows_min
        move = float(window["high"].max()) - last_close

    if displacement and move >= min_move:
        reasons.append(
            f"M15: desplazamiento {'alcista' if direction == 'long' else 'bajista'} "
            f"confirmado (move={move:.5f}) ✓"
        )
        return True, reasons

    reasons.append(
        f"M15: desplazamiento insuficiente (move={move:.5f}, min={min_move:.5f})"
    )
    return False, reasons


# ---------------------------------------------------------------------------
# Sistema de puntuación LCC
# ---------------------------------------------------------------------------

def _compute_score(h4_valid: bool, h1_valid: bool, m15_valid: bool) -> int:
    """
    Puntuación sobre 10. Umbral mínimo para operar: MIN_SCORE (8).
      H4 contexto macro      → +3  (L: zona de liquidez institucional)
      H1 liquidez+cobertura  → +4  (L+C: regla crítica, pondera más)
      M15 confirmación       → +3  (C: validación fractal)
    """
    score = 0
    score += 3 if h4_valid else 0
    score += 4 if h1_valid else 0
    score += 3 if m15_valid else 0
    return score


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _find_swing_high(df: pd.DataFrame, lookback: int) -> float | None:
    """Máximo relevante en las últimas `lookback` velas."""
    if len(df) < lookback:
        return None
    return float(df.tail(lookback)["high"].max())


def _find_swing_low(df: pd.DataFrame, lookback: int) -> float | None:
    """Mínimo relevante en las últimas `lookback` velas."""
    if len(df) < lookback:
        return None
    return float(df.tail(lookback)["low"].min())


def _resample_to_m15(h1: pd.DataFrame) -> pd.DataFrame:
    """
    Resamplea H1 a M15. Requiere DatetimeIndex.
    Si el índice no es datetime (ej. en tests con índice entero),
    retorna el dataframe original como fallback.
    """
    try:
        return h1.resample("15min").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna()
    except Exception:
        return h1


def _fallback_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR de fallback cuando entry_df no tiene atr_14 precalculado."""
    series = calculate_atr(df, period=period)
    val = series.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


def is_valid_session() -> bool:
    """Permite entradas solo en London killzone y solapamiento London-NY (UTC)."""
    current_hour = datetime.now(timezone.utc).hour
    london_session = 7 <= current_hour < 10
    new_york_session = 13 <= current_hour < 16
    return london_session or new_york_session