import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
from src.utils.logger import get_logger
from src.strategy.chart_patterns import ChartPatternEngine, ChartPatternType, PricingZone

logger = get_logger(__name__)

@dataclass
class StructureAnalysis:
    bias_15m: str
    bos_5m: bool
    choch_5m: bool
    bos_direction_5m: str
    liquidity_sweep_5m: bool
    displacement_1m: bool
    retest_1m: bool
    is_valid: bool
    invalidation_reason: Optional[str]
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    support_levels: list = None
    resistance_levels: list = None
    setup_type: str = "NONE"
    sweep_direction: str = "NONE"
    at_key_level: bool = False
    fvg_detected: bool = False
    fvg_direction: str = "NONE"
    fvg_top: float = 0.0
    fvg_bottom: float = 0.0
    fvg_testing: bool = False
    pdh: float = 0.0
    pdl: float = 0.0
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    eqh: float = 0.0
    eql: float = 0.0
    asian_high: float = 0.0
    asian_low: float = 0.0
    ob_detected: bool = False
    ob_direction: str = "NONE"
    ob_top: float = 0.0
    ob_bottom: float = 0.0
    ob_testing: bool = False
    is_squeeze: bool = False
    squeeze_fired: bool = False
    breakout_level: float = 0.0
    breakout_direction: str = "NONE"
    pwh: float = 0.0
    pwl: float = 0.0
    is_bull_trap: bool = False
    is_bear_trap: bool = False
    is_judas_swing: bool = False
    is_volume_absorption: bool = False
    trap_level: float = 0.0
    trap_wick_extreme: float = 0.0
    chart_pattern: str = "NONE"
    chart_pattern_direction: str = "NEUTRAL"
    chart_pattern_conf: float = 0.0
    chart_pattern_level: float = 0.0
    chart_pattern_target: float = 0.0
    pricing_zone: str = "EQUILIBRIUM"
    range_position_pct: float = 50.0
    playbook: str = "NONE"

    def __post_init__(self):
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []


def calculate_volatility_squeeze(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """John Carter Volatility Squeeze indicator.
    
    Bollinger Bands (period, bb_mult) vs Keltner Channels (period, kc_mult * ATR).
    - is_squeeze: True when BB is entirely inside KC (Upper BB < Upper KC and Lower BB > Lower KC)
    - squeeze_fired: True when previous bar was in squeeze and current bar is no longer squeezed
    Returns: (is_squeeze: np.ndarray[bool], squeeze_fired: np.ndarray[bool])
    """
    n = len(close)
    is_squeeze = np.zeros(n, dtype=bool)
    squeeze_fired = np.zeros(n, dtype=bool)
    period = max(bb_period, kc_period)
    if n < period:
        return is_squeeze, squeeze_fired

    pad = bb_period - 1
    sma = np.convolve(close, np.ones(bb_period) / bb_period, mode='valid')

    stds = np.array([np.std(close[i - pad : i + 1]) for i in range(pad, n)])
    upper_bb = sma + bb_mult * stds
    lower_bb = sma - bb_mult * stds

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (kc_period - 1) + tr[i]) / kc_period

    upper_kc = sma + kc_mult * atr[pad:]
    lower_kc = sma - kc_mult * atr[pad:]

    squeeze_slice = (upper_bb < upper_kc) & (lower_bb > lower_kc)
    is_squeeze[pad:] = squeeze_slice

    for i in range(pad + 1, n):
        if is_squeeze[i - 1] and not is_squeeze[i]:
            squeeze_fired[i] = True

    return is_squeeze, squeeze_fired


def compute_volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    num_bins: int = 30,
    value_area_pct: float = 0.70,
) -> Tuple[float, float, float]:
    """Calculate Volume Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    
    Returns: (poc, vah, val)
    """
    if len(close) == 0 or len(volume) == 0:
        return 0.0, 0.0, 0.0
    min_p = float(np.min(low))
    max_p = float(np.max(high))
    if min_p >= max_p:
        c = float(close[-1])
        return c, c, c

    bin_edges = np.linspace(min_p, max_p, num_bins + 1)
    bin_volumes = np.zeros(num_bins, dtype=float)
    candle_mids = (high + low + close) / 3.0

    for mid, vol in zip(candle_mids, volume):
        b_idx = int((mid - min_p) / (max_p - min_p) * num_bins)
        b_idx = min(max(b_idx, 0), num_bins - 1)
        bin_volumes[b_idx] += float(vol)

    max_idx = int(np.argmax(bin_volumes))
    poc = float((bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2.0)

    # 70% Value Area calculation
    total_vol = float(np.sum(bin_volumes))
    if total_vol <= 0:
        return poc, float(max_p), float(min_p)

    target_vol = total_vol * value_area_pct
    current_vol = float(bin_volumes[max_idx])
    lower_idx = max_idx
    upper_idx = max_idx

    while current_vol < target_vol and (lower_idx > 0 or upper_idx < num_bins - 1):
        vol_above = float(bin_volumes[upper_idx + 1]) if upper_idx < num_bins - 1 else 0.0
        vol_below = float(bin_volumes[lower_idx - 1]) if lower_idx > 0 else 0.0

        if vol_above >= vol_below and upper_idx < num_bins - 1:
            upper_idx += 1
            current_vol += vol_above
        elif lower_idx > 0:
            lower_idx -= 1
            current_vol += vol_below
        elif upper_idx < num_bins - 1:
            upper_idx += 1
            current_vol += vol_above
        else:
            break

    val = float(bin_edges[lower_idx])
    vah = float(bin_edges[upper_idx + 1])
    return poc, vah, val


def compute_volume_poc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    num_bins: int = 30,
) -> float:
    """Calculate Volume Point of Control (POC) across given price/volume arrays."""
    poc, _, _ = compute_volume_profile(high, low, close, volume, num_bins=num_bins)
    return poc


def detect_equal_highs_lows(
    high: np.ndarray,
    low: np.ndarray,
    swing_highs: List[int],
    swing_lows: List[int],
    tolerance_pct: float = 0.0005,
) -> Tuple[float, float]:
    """Detect Equal Highs (EQH - Buy-Side Liquidity) and Equal Lows (EQL - Sell-Side Liquidity).
    
    Returns: (eqh_price, eql_price)
    """
    eqh = 0.0
    eql = 0.0

    # Scan pairs of recent swing highs
    if len(swing_highs) >= 2:
        for i in range(len(swing_highs) - 1, 0, -1):
            h1 = float(high[swing_highs[i]])
            h2 = float(high[swing_highs[i - 1]])
            if h1 > 0 and abs(h1 - h2) / h1 <= tolerance_pct:
                eqh = (h1 + h2) / 2.0
                break

    # Scan pairs of recent swing lows
    if len(swing_lows) >= 2:
        for i in range(len(swing_lows) - 1, 0, -1):
            l1 = float(low[swing_lows[i]])
            l2 = float(low[swing_lows[i - 1]])
            if l1 > 0 and abs(l1 - l2) / l1 <= tolerance_pct:
                eql = (l1 + l2) / 2.0
                break

    return eqh, eql


def detect_order_blocks(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    lookback: int = 15,
    displacement_threshold: float = 0.6,
) -> Tuple[bool, str, float, float, bool]:
    """Detect Institutional Order Block (OB) and evaluate current price testing/mitigation.
    
    Definition:
    - Bullish OB: Last down candle (close < open) prior to a bullish displacement candle.
    - Bearish OB: Last up candle (close > open) prior to a bearish displacement candle.
    
    Returns: (ob_detected, ob_direction, ob_top, ob_bottom, ob_testing)
    """
    n = len(close_arr)
    if n < 4:
        return False, "NONE", 0.0, 0.0, False

    cur_price = float(close_arr[-1])
    scan_limit = min(lookback, n - 2)

    for offset in range(1, scan_limit + 1):
        i = n - offset
        if i < 2:
            break

        # Check for displacement move on candle i
        body = abs(close_arr[i] - open_arr[i])
        rng = high_arr[i] - low_arr[i]
        if rng <= 0 or (body / rng) < displacement_threshold:
            continue

        # Bullish displacement: green candle with large body breaking prior high
        if close_arr[i] > open_arr[i] and close_arr[i] > high_arr[i - 1]:
            for j in range(i - 1, max(-1, i - 4), -1):
                if close_arr[j] <= open_arr[j]:  # Bearish origin candle
                    ob_top = float(high_arr[j])
                    ob_bottom = float(low_arr[j])
                    broken = np.any(close_arr[i:] < ob_bottom)
                    if not broken:
                        is_testing = (ob_bottom <= cur_price <= ob_top) or (low_arr[-1] <= ob_top and cur_price >= ob_bottom and cur_price <= ob_top * 1.005)
                        return True, "BULLISH", ob_top, ob_bottom, bool(is_testing)

        # Bearish displacement: red candle with large body breaking prior low
        elif close_arr[i] < open_arr[i] and close_arr[i] < low_arr[i - 1]:
            for j in range(i - 1, max(-1, i - 4), -1):
                if close_arr[j] >= open_arr[j]:  # Bullish origin candle
                    ob_top = float(high_arr[j])
                    ob_bottom = float(low_arr[j])
                    broken = np.any(close_arr[i:] > ob_top)
                    if not broken:
                        is_testing = (ob_bottom <= cur_price <= ob_top) or (high_arr[-1] >= ob_bottom and cur_price <= ob_top and cur_price >= ob_bottom * 0.995)
                        return True, "BEARISH", ob_top, ob_bottom, bool(is_testing)

    return False, "NONE", 0.0, 0.0, False


def calculate_asian_range(candles: Any) -> Tuple[float, float]:
    """Calculate Asian Session (00:00 - 08:00 UTC) High and Low.
    
    Returns: (asian_high, asian_low)
    """
    if not candles:
        return 0.0, 0.0

    asian_highs = []
    asian_lows = []

    for c in candles:
        open_time = getattr(c, "open_time", None) if not isinstance(c, dict) else c.get("open_time")
        hi = getattr(c, "high", None) if not isinstance(c, dict) else c.get("high")
        lo = getattr(c, "low", None) if not isinstance(c, dict) else c.get("low")
        if open_time and hi is not None and lo is not None:
            try:
                from datetime import datetime, timezone
                ts = float(open_time) / 1000.0 if float(open_time) > 1e11 else float(open_time)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                if 0 <= dt.hour < 8:
                    asian_highs.append(float(hi))
                    asian_lows.append(float(lo))
            except Exception:
                pass

    if asian_highs and asian_lows:
        return float(np.max(asian_highs)), float(np.min(asian_lows))

    if len(candles) >= 32:
        slice_32 = candles[:32]
        his = [c.high if not isinstance(c, dict) else c['high'] for c in slice_32]
        los = [c.low if not isinstance(c, dict) else c['low'] for c in slice_32]
        return float(np.max(his)), float(np.min(los))

    return 0.0, 0.0


def detect_trap_market(
    op: np.ndarray,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    vol: np.ndarray,
    key_levels: Dict[str, float],
    atr: float = 0.0,
    wick_ratio_threshold: float = 0.20,
    cur_timestamp_ms: Optional[int] = None,
) -> Tuple[bool, bool, bool, bool, str, float, float]:
    """Detect institutional trap market patterns:
    1. Bull Trap / Buy-Side SFP:
       Price breaches an upper key level (Resistance, EQH, PDH, PWH, VAH),
       then violently rejects and closes below it with an upper wick.
    2. Bear Trap / Sell-Side SFP:
       Price plunges below a lower key level (Support, EQL, PDL, PWL, VAL),
       then snaps back and closes above it with a lower wick.
    3. Judas Swing:
       Session open manipulation (London Open 07:00-09:00 UTC or NY Open 13:00-15:00 UTC)
       sweeping Asian High or Asian Low and reversing into the range.
    4. Volume Absorption:
       High volume spike (>= 2.2x SMA20) on compressed body (<= 0.35 range) at a key level.

    Returns:
        (is_bull_trap, is_bear_trap, is_judas_swing, is_volume_absorption,
         trap_type, trap_level, trap_wick_extreme)
    """
    n = len(cl)
    if n < 2:
        return False, False, False, False, "NONE", 0.0, 0.0

    cur_o, cur_h, cur_l, cur_c = op[-1], hi[-1], lo[-1], cl[-1]
    rng = cur_h - cur_l
    if rng <= 0:
        return False, False, False, False, "NONE", 0.0, 0.0

    upper_wick = cur_h - max(cur_o, cur_c)
    lower_wick = min(cur_o, cur_c) - cur_l
    body = abs(cur_c - cur_o)

    upper_wick_ratio = upper_wick / rng
    lower_wick_ratio = lower_wick / rng

    # Check for session manipulation window (Judas Swing)
    is_session_window = False
    if cur_timestamp_ms:
        try:
            from datetime import datetime, timezone
            ts = float(cur_timestamp_ms) / 1000.0 if float(cur_timestamp_ms) > 1e11 else float(cur_timestamp_ms)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if (7 <= dt.hour <= 9) or (12 <= dt.hour <= 15):
                is_session_window = True
        except Exception:
            pass

    # Upper key levels for Bull Trap / Buy-side liquidity hunt
    upper_levels = [
        ("PWH", key_levels.get("PWH", 0.0)),
        ("PDH", key_levels.get("PDH", 0.0)),
        ("EQH", key_levels.get("EQH", 0.0)),
        ("RESISTANCE", key_levels.get("NEAREST_RESISTANCE", 0.0)),
        ("VAH", key_levels.get("VAH", 0.0)),
        ("ASIAN_HIGH", key_levels.get("ASIAN_HIGH", 0.0)),
    ]

    # Lower key levels for Bear Trap / Sell-side liquidity hunt
    lower_levels = [
        ("PWL", key_levels.get("PWL", 0.0)),
        ("PDL", key_levels.get("PDL", 0.0)),
        ("EQL", key_levels.get("EQL", 0.0)),
        ("SUPPORT", key_levels.get("NEAREST_SUPPORT", 0.0)),
        ("VAL", key_levels.get("VAL", 0.0)),
        ("ASIAN_LOW", key_levels.get("ASIAN_LOW", 0.0)),
    ]

    # 1. Bull Trap Detection (Buy-side SFP)
    for name, lvl in upper_levels:
        if lvl and lvl > 0:
            if cur_h > lvl and cur_c < lvl and cur_h > hi[-2]:
                if upper_wick_ratio >= wick_ratio_threshold or cur_c <= cur_o:
                    is_judas = (name == "ASIAN_HIGH" and is_session_window)
                    trap_type = "JUDAS_SWING_HIGH" if is_judas else f"BULL_TRAP_{name}"
                    return True, False, is_judas, False, trap_type, float(lvl), float(cur_h)

    # 2. Bear Trap Detection (Sell-side SFP)
    for name, lvl in lower_levels:
        if lvl and lvl > 0:
            if cur_l < lvl and cur_c > lvl and cur_l < lo[-2]:
                if lower_wick_ratio >= wick_ratio_threshold or cur_c >= cur_o:
                    is_judas = (name == "ASIAN_LOW" and is_session_window)
                    trap_type = "JUDAS_SWING_LOW" if is_judas else f"BEAR_TRAP_{name}"
                    return False, True, is_judas, False, trap_type, float(lvl), float(cur_l)

    # 3. Volume Absorption Detection
    if len(vol) >= 20:
        avg_vol = float(np.mean(vol[-20:]))
        rel_vol = (vol[-1] / avg_vol) if avg_vol > 0 else 1.0
        if rel_vol >= 2.2 and (body / rng) <= 0.35:
            if upper_wick_ratio >= 0.40:
                return True, False, False, True, "VOLUME_ABSORPTION_HIGH", float(cur_h), float(cur_h)
            elif lower_wick_ratio >= 0.40:
                return False, True, False, True, "VOLUME_ABSORPTION_LOW", float(cur_l), float(cur_l)

    return False, False, False, False, "NONE", 0.0, 0.0


class MarketStructure:
    def __init__(
        self, 
        config: Optional[Any] = None,
        lookback: int = 10, 
        wick_ratio_threshold: float = 0.6, 
        displacement_threshold: float = 0.7, 
        retest_tolerance_mult: float = 0.5,
        fvg_min_atr_mult: float = 0.3,
    ):
        if config is not None and not isinstance(config, int):
            self.lookback = getattr(config, "min_swing_lookback", getattr(config, "lookback", 10))
            self.wick_ratio_threshold = getattr(config, "liquidity_sweep_wick_ratio", getattr(config, "wick_ratio_threshold", 0.6))
            self.displacement_threshold = getattr(config, "displacement_body_ratio", getattr(config, "displacement_threshold", 0.7))
            self.retest_tolerance_mult = getattr(config, "retest_tolerance_atr_mult", getattr(config, "retest_tolerance_mult", 0.5))
            self.fvg_min_atr_mult = getattr(config, "fvg_min_atr_mult", fvg_min_atr_mult)
            breakout_cfg = getattr(config, "breakout", None)
            self.squeeze_bb_mult = getattr(breakout_cfg, "squeeze_bb_mult", 2.0) if breakout_cfg else 2.0
            self.squeeze_kc_mult = getattr(breakout_cfg, "squeeze_kc_mult", 1.5) if breakout_cfg else 1.5
            self.volume_expansion_mult = getattr(breakout_cfg, "volume_expansion_mult", 2.0) if breakout_cfg else 2.0
            self.displacement_body_pct = getattr(breakout_cfg, "displacement_body_pct", 0.70) if breakout_cfg else 0.70
        else:
            self.lookback = config if isinstance(config, int) else lookback
            self.wick_ratio_threshold = wick_ratio_threshold
            self.displacement_threshold = displacement_threshold
            self.retest_tolerance_mult = retest_tolerance_mult
            self.fvg_min_atr_mult = fvg_min_atr_mult
            self.squeeze_bb_mult = 2.0
            self.squeeze_kc_mult = 1.5
            self.volume_expansion_mult = 2.0
            self.displacement_body_pct = 0.70

    def _extract_arrays(self, candles: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not candles:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        if isinstance(candles, dict) and "open" in candles:
            return (
                np.asarray(candles["open"]),
                np.asarray(candles["high"]),
                np.asarray(candles["low"]),
                np.asarray(candles["close"]),
                np.asarray(candles["volume"]),
            )
        
        if isinstance(candles[0], dict):
            op = np.array([c['open'] for c in candles])
            hi = np.array([c['high'] for c in candles])
            lo = np.array([c['low'] for c in candles])
            cl = np.array([c['close'] for c in candles])
            vol = np.array([c['volume'] for c in candles])
        else:
            op = np.array([c.open for c in candles])
            hi = np.array([c.high for c in candles])
            lo = np.array([c.low for c in candles])
            cl = np.array([c.close for c in candles])
            vol = np.array([c.volume for c in candles])
        return op, hi, lo, cl, vol

    def find_swing_points(self, highs: np.ndarray, lows: np.ndarray, lookback: Optional[int] = None) -> Tuple[List[int], List[int]]:
        lb = lookback if lookback is not None else self.lookback
        n = len(highs)
        swing_highs = []
        swing_lows = []
        for i in range(lb, n - lb):
            if np.all(highs[i] > highs[i - lb : i]) and np.all(highs[i] > highs[i + 1 : i + lb + 1]):
                swing_highs.append(i)
            if np.all(lows[i] < lows[i - lb : i]) and np.all(lows[i] < lows[i + 1 : i + lb + 1]):
                swing_lows.append(i)
        return swing_highs, swing_lows

    def get_support_resistance_levels(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> tuple:
        """Extract key support and resistance levels from swing points.
        
        Returns: (support_levels, resistance_levels, nearest_support, nearest_resistance)
        """
        swing_highs, swing_lows = self.find_swing_points(highs, lows)
        current_price = closes[-1] if len(closes) > 0 else 0.0
        
        # Get resistance levels (swing highs above current price)
        resistance_levels = sorted(
            [float(highs[i]) for i in swing_highs if highs[i] > current_price]
        )
        # Get support levels (swing lows below current price)
        support_levels = sorted(
            [float(lows[i]) for i in swing_lows if lows[i] < current_price],
            reverse=True
        )
        
        nearest_support = support_levels[0] if support_levels else 0.0
        nearest_resistance = resistance_levels[0] if resistance_levels else 0.0
        
        return support_levels, resistance_levels, nearest_support, nearest_resistance

    def detect_fair_value_gaps(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr: float = 0.0
    ) -> Tuple[bool, str, float, float, bool]:
        """Detect 3-candle Fair Value Gap (FVG) and determine if current price is testing/mitigating it.
        
        Definition:
        - Bullish FVG: Low[i] > High[i-2] (gap between candle 1 high and candle 3 low).
        - Bearish FVG: High[i] < Low[i-2] (gap between candle 1 low and candle 3 high).
        
        Returns: (fvg_detected, fvg_direction, fvg_top, fvg_bottom, is_testing)
        """
        n = len(highs)
        if n < 3:
            return False, "NONE", 0.0, 0.0, False

        min_gap = (atr * self.fvg_min_atr_mult) if atr > 0 else 0.0
        cur_price = closes[-1] if len(closes) > 0 else 0.0

        scan_limit = min(15, n - 2)
        for offset in range(1, scan_limit + 1):
            i = n - offset
            if i < 2:
                break

            c1_high = highs[i - 2]
            c1_low = lows[i - 2]
            c3_high = highs[i]
            c3_low = lows[i]

            # Bullish FVG: c3_low > c1_high
            if c3_low > c1_high:
                gap = c3_low - c1_high
                if gap >= min_gap:
                    fvg_top = float(c3_low)
                    fvg_bottom = float(c1_high)
                    subsequent_lows = lows[i:]
                    is_testing = any(fvg_bottom <= l_val <= fvg_top or fvg_bottom <= cur_price <= fvg_top for l_val in subsequent_lows)
                    if cur_price >= fvg_bottom:
                        return True, "BULLISH", fvg_top, fvg_bottom, is_testing

            # Bearish FVG: c3_high < c1_low
            elif c3_high < c1_low:
                gap = c1_low - c3_high
                if gap >= min_gap:
                    fvg_top = float(c1_low)
                    fvg_bottom = float(c3_high)
                    subsequent_highs = highs[i:]
                    is_testing = any(fvg_bottom <= h_val <= fvg_top or fvg_bottom <= cur_price <= fvg_top for h_val in subsequent_highs)
                    if cur_price <= fvg_top:
                        return True, "BEARISH", fvg_top, fvg_bottom, is_testing

        return False, "NONE", 0.0, 0.0, False

    def analyze(
        self,
        candles_15m: List[Any],
        candles_5m: List[Any],
        candles_1m: List[Any],
        atr_1m: float,
        htf_levels: Optional[Any] = None,
    ) -> StructureAnalysis:
        logger.info("Analyzing market structure across timeframes")
        
        # 15m analysis
        op_15, hi_15, lo_15, cl_15, vol_15 = self._extract_arrays(candles_15m)

        # Higher-Timeframe (HTF) S/R, Volume Profile & Asian Range Levels
        pwh = float(htf_levels.get("PWH", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        pwl = float(htf_levels.get("PWL", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        pdh = float(htf_levels.get("PDH", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        pdl = float(htf_levels.get("PDL", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        poc = float(htf_levels.get("POC", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        vah = float(htf_levels.get("VAH", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        val = float(htf_levels.get("VAL", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        asian_high = float(htf_levels.get("ASIAN_HIGH", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0
        asian_low = float(htf_levels.get("ASIAN_LOW", 0.0)) if (htf_levels and isinstance(htf_levels, dict)) else 0.0

        if (poc == 0.0 or vah == 0.0 or val == 0.0) and len(cl_15) >= 20 and len(vol_15) >= 20:
            calc_poc, calc_vah, calc_val = compute_volume_profile(hi_15, lo_15, cl_15, vol_15)
            if poc == 0.0:
                poc = calc_poc
            if vah == 0.0:
                vah = calc_vah
            if val == 0.0:
                val = calc_val

        if (pdh == 0.0 or pdl == 0.0) and len(cl_15) >= 20:
            pdh = float(np.max(hi_15))
            pdl = float(np.min(lo_15))

        if asian_high == 0.0 and asian_low == 0.0:
            calc_ah, calc_al = calculate_asian_range(candles_15m)
            if calc_ah == 0.0 and calc_al == 0.0:
                calc_ah, calc_al = calculate_asian_range(candles_5m)
            if asian_high == 0.0:
                asian_high = calc_ah
            if asian_low == 0.0:
                asian_low = calc_al

        bias_15m = 'NEUTRAL'
        lookback_15 = min(self.lookback, 4)
        if len(hi_15) > 2 * lookback_15:
            sh_15, sl_15 = self.find_swing_points(hi_15, lo_15, lookback=lookback_15)
            
            # 1. Progression of swing highs / lows
            if len(sh_15) >= 2 and hi_15[sh_15[-1]] > hi_15[sh_15[-2]]:
                bias_15m = 'BULLISH'
            elif len(sl_15) >= 2 and lo_15[sl_15[-1]] < lo_15[sl_15[-2]]:
                bias_15m = 'BEARISH'

            # 2. Check breaks of swing levels
            if sh_15:
                last_sh_price = hi_15[sh_15[-1]]
                if cl_15[-1] > last_sh_price:
                    bias_15m = 'BULLISH'
                elif sh_15[-1] < len(hi_15) - 1 and np.max(hi_15[sh_15[-1] + 1:]) > last_sh_price:
                    bias_15m = 'BULLISH'

            if sl_15:
                last_sl_price = lo_15[sl_15[-1]]
                if cl_15[-1] < last_sl_price:
                    bias_15m = 'BEARISH'
                elif sl_15[-1] < len(lo_15) - 1 and np.min(lo_15[sl_15[-1] + 1:]) < last_sl_price:
                    bias_15m = 'BEARISH'

            # 3. If both exist, resolve conflicting signals by checking which break happened most recently
            if sh_15 and sl_15:
                last_sh_idx = sh_15[-1]
                last_sl_idx = sl_15[-1]
                last_sh_price = hi_15[last_sh_idx]
                last_sl_price = lo_15[last_sl_idx]

                broke_high = (last_sh_idx < len(hi_15) - 1 and np.max(hi_15[last_sh_idx + 1:]) > last_sh_price) or (cl_15[-1] > last_sh_price)
                broke_low = (last_sl_idx < len(lo_15) - 1 and np.min(lo_15[last_sl_idx + 1:]) < last_sl_price) or (cl_15[-1] < last_sl_price)

                if broke_high and not broke_low:
                    bias_15m = 'BULLISH'
                elif broke_low and not broke_high:
                    bias_15m = 'BEARISH'
                elif len(sh_15) >= 2 and len(sl_15) >= 2:
                    if hi_15[sh_15[-1]] > hi_15[sh_15[-2]] and lo_15[sl_15[-1]] > lo_15[sl_15[-2]]:
                        bias_15m = 'BULLISH'
                    elif hi_15[sh_15[-1]] < hi_15[sh_15[-2]] and lo_15[sl_15[-1]] < lo_15[sl_15[-2]]:
                        bias_15m = 'BEARISH'

            # 4. Fallback to 15m trend & moving average if swing structure is neutral
            if bias_15m == 'NEUTRAL' and len(cl_15) >= 20:
                from src.strategy.signals import compute_ema
                ema_20_15 = compute_ema(cl_15, 20)
                ema_50_15 = compute_ema(cl_15, min(50, len(cl_15)))
                if cl_15[-1] > ema_20_15[-1] and ema_20_15[-1] >= ema_50_15[-1]:
                    bias_15m = 'BULLISH'
                elif cl_15[-1] < ema_20_15[-1] and ema_20_15[-1] <= ema_50_15[-1]:
                    bias_15m = 'BEARISH'
                elif cl_15[-1] > ema_20_15[-1] and cl_15[-1] > cl_15[-min(5, len(cl_15))]:
                    bias_15m = 'BULLISH'
                elif cl_15[-1] < ema_20_15[-1] and cl_15[-1] < cl_15[-min(5, len(cl_15))]:
                    bias_15m = 'BEARISH'
        
        # 5m analysis
        op_5, hi_5, lo_5, cl_5, vol_5 = self._extract_arrays(candles_5m)
        bos_5m = False
        choch_5m = False
        bos_direction_5m = 'NONE'
        liquidity_sweep_5m = False
        broken_level_5m: Optional[float] = None
        eqh, eql = 0.0, 0.0
        
        if len(hi_5) > 2 * self.lookback:
            sh_5, sl_5 = self.find_swing_points(hi_5, lo_5)
            eqh, eql = detect_equal_highs_lows(hi_5, lo_5, sh_5, sl_5)
            
            # Check Break of Structure and CHoCH
            if sh_5 and cl_5[-1] > hi_5[sh_5[-1]]:
                bos_5m = True
                bos_direction_5m = 'BULLISH'
                broken_level_5m = hi_5[sh_5[-1]]
                # CHoCH: break above last swing high while previous swing structure was in a downtrend
                if len(sh_5) >= 2 and hi_5[sh_5[-1]] < hi_5[sh_5[-2]]:
                    choch_5m = True
            elif sl_5 and cl_5[-1] < lo_5[sl_5[-1]]:
                bos_5m = True
                bos_direction_5m = 'BEARISH'
                broken_level_5m = lo_5[sl_5[-1]]
                # CHoCH: break below last swing low while previous swing structure was in an uptrend
                if len(sl_5) >= 2 and lo_5[sl_5[-1]] > lo_5[sl_5[-2]]:
                    choch_5m = True
                
            # Liquidity sweep check
            if bos_direction_5m == 'NONE':
                if sh_5:
                    last_sh = sh_5[-1]
                    wick_beyond = hi_5[-1] - hi_5[last_sh]
                    total_range = hi_5[-1] - lo_5[-1]
                    if total_range > 0 and wick_beyond > 0 and cl_5[-1] < hi_5[last_sh]:
                        if (wick_beyond / total_range) > self.wick_ratio_threshold:
                            liquidity_sweep_5m = True
                if sl_5:
                    last_sl = sl_5[-1]
                    wick_beyond = lo_5[last_sl] - lo_5[-1]
                    total_range = hi_5[-1] - lo_5[-1]
                    if total_range > 0 and wick_beyond > 0 and cl_5[-1] > lo_5[last_sl]:
                        if (wick_beyond / total_range) > self.wick_ratio_threshold:
                            liquidity_sweep_5m = True

        # Order Block detection (5m primary, 1m fallback)
        ob_detected, ob_direction, ob_top, ob_bottom, ob_testing = detect_order_blocks(op_5, hi_5, lo_5, cl_5)

        # 1m analysis
        op_1, hi_1, lo_1, cl_1, vol_1 = self._extract_arrays(candles_1m)
        displacement_1m = False
        retest_1m = False

        if not ob_detected and len(cl_1) >= 4:
            ob_detected, ob_direction, ob_top, ob_bottom, ob_testing = detect_order_blocks(op_1, hi_1, lo_1, cl_1)
        
        if len(cl_1) > 0:
            body = abs(cl_1[-1] - op_1[-1])
            rng = hi_1[-1] - lo_1[-1]
            if rng > 0 and (body / rng) > self.displacement_threshold:
                displacement_1m = True
                
            # Retest detection against broken 5m swing level
            if broken_level_5m is not None and atr_1m > 0:
                tolerance = atr_1m * self.retest_tolerance_mult
                # Check recent 1m candles for touch or pullback to broken level
                recent_lo = lo_1[-min(5, len(lo_1)):]
                recent_hi = hi_1[-min(5, len(hi_1)):]
                recent_cl = cl_1[-min(5, len(cl_1)):]
                for l_val, h_val, c_val in zip(recent_lo, recent_hi, recent_cl):
                    if abs(c_val - broken_level_5m) <= tolerance or (l_val <= broken_level_5m <= h_val):
                        retest_1m = True
                        break

        # Validity & confluence enforcement
        is_valid = True
        invalidation_reason = None
        
        if bias_15m == 'NEUTRAL':
            is_valid = False
            invalidation_reason = "15m Bias is NEUTRAL"
        elif bias_15m == 'BULLISH' and bos_direction_5m == 'BEARISH':
            is_valid = False
            invalidation_reason = "5m Bearish BOS contradicts Bullish 15m bias"
        elif bias_15m == 'BEARISH' and bos_direction_5m == 'BULLISH':
            is_valid = False
            invalidation_reason = "5m Bullish BOS contradicts Bearish 15m bias"
        
        # Calculate support/resistance from 5m swing points
        nearest_support = 0.0
        nearest_resistance = 0.0
        support_levels_list = []
        resistance_levels_list = []
        if len(hi_5) > 2 * self.lookback:
            support_levels_list, resistance_levels_list, nearest_support, nearest_resistance = (
                self.get_support_resistance_levels(hi_5, lo_5, cl_5)
            )

        # Detect sweep direction and key level proximity
        at_key_level = False
        sweep_direction = 'NONE'
        if len(cl_1) > 0:
            cur_p = cl_1[-1]
            if nearest_support > 0 and abs(cur_p - nearest_support) <= (atr_1m * 1.5):
                at_key_level = True
            if nearest_resistance > 0 and abs(cur_p - nearest_resistance) <= (atr_1m * 1.5):
                at_key_level = True

            # Micro 1m Liquidity Sweep at key support (Bullish sweep reversal)
            if nearest_support > 0 and lo_1[-1] < nearest_support and cl_1[-1] > nearest_support:
                wick = nearest_support - lo_1[-1]
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.30:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BULLISH'

            # Micro 1m Liquidity Sweep at key resistance (Bearish sweep reversal)
            if nearest_resistance > 0 and hi_1[-1] > nearest_resistance and cl_1[-1] < nearest_resistance:
                wick = hi_1[-1] - nearest_resistance
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.30:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BEARISH'

            # Sweep of Equal Highs (EQH - Buy-Side Liquidity hunt)
            if eqh > 0 and hi_1[-1] > eqh and cl_1[-1] < eqh:
                wick = hi_1[-1] - eqh
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.20:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BEARISH'

            # Sweep of Equal Lows (EQL - Sell-Side Liquidity hunt)
            if eql > 0 and lo_1[-1] < eql and cl_1[-1] > eql:
                wick = eql - lo_1[-1]
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.20:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BULLISH'

            # Sweep of Asian High / Low
            if asian_high > 0 and hi_1[-1] > asian_high and cl_1[-1] < asian_high:
                wick = hi_1[-1] - asian_high
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.20:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BEARISH'
            elif asian_low > 0 and lo_1[-1] < asian_low and cl_1[-1] > asian_low:
                wick = asian_low - lo_1[-1]
                rng_1 = hi_1[-1] - lo_1[-1]
                if rng_1 > 0 and (wick / rng_1) >= 0.20:
                    liquidity_sweep_5m = True
                    sweep_direction = 'BULLISH'

        # Also fallback sweep checks from 5m candle if 1m didn't trigger
        if sweep_direction == 'NONE':
            if eqh > 0 and len(hi_5) > 0 and hi_5[-1] > eqh and cl_5[-1] < eqh:
                liquidity_sweep_5m = True
                sweep_direction = 'BEARISH'
            elif eql > 0 and len(lo_5) > 0 and lo_5[-1] < eql and cl_5[-1] > eql:
                liquidity_sweep_5m = True
                sweep_direction = 'BULLISH'
            elif asian_high > 0 and len(hi_5) > 0 and hi_5[-1] > asian_high and cl_5[-1] < asian_high:
                liquidity_sweep_5m = True
                sweep_direction = 'BEARISH'
            elif asian_low > 0 and len(lo_5) > 0 and lo_5[-1] < asian_low and cl_5[-1] > asian_low:
                liquidity_sweep_5m = True
                sweep_direction = 'BULLISH'

        # Also from 5m liquidity sweep:
        if sweep_direction == 'NONE' and liquidity_sweep_5m:
            if sh_5 and hi_5[-1] > hi_5[sh_5[-1]]:
                sweep_direction = 'BEARISH'
            elif sl_5 and lo_5[-1] < lo_5[sl_5[-1]]:
                sweep_direction = 'BULLISH'

        # Fair Value Gap detection (check 1m first, fallback to 5m)
        fvg_detected = False
        fvg_dir = "NONE"
        fvg_top = 0.0
        fvg_bottom = 0.0
        fvg_testing = False

        if len(hi_1) >= 3:
            fvg_detected, fvg_dir, fvg_top, fvg_bottom, fvg_testing = self.detect_fair_value_gaps(
                hi_1, lo_1, cl_1, atr=atr_1m
            )
        if not fvg_detected and len(hi_5) >= 3:
            fvg_detected, fvg_dir, fvg_top, fvg_bottom, fvg_testing = self.detect_fair_value_gaps(
                hi_5, lo_5, cl_5, atr=atr_1m * 2.0
            )

        # Volatility Squeeze detection
        is_squeeze = False
        squeeze_fired = False
        if len(cl_5) >= 20:
            is_sq_5, sq_fired_5 = calculate_volatility_squeeze(
                hi_5, lo_5, cl_5, bb_mult=self.squeeze_bb_mult, kc_mult=self.squeeze_kc_mult
            )
            is_squeeze = bool(is_sq_5[-1])
            squeeze_fired = bool(sq_fired_5[-1])
        elif len(cl_1) >= 20:
            is_sq_1, sq_fired_1 = calculate_volatility_squeeze(
                hi_1, lo_1, cl_1, bb_mult=self.squeeze_bb_mult, kc_mult=self.squeeze_kc_mult
            )
            is_squeeze = bool(is_sq_1[-1])
            squeeze_fired = bool(sq_fired_1[-1])

        # Integrate POC into key S/R levels
        if poc > 0.0:
            if len(cl_1) > 0:
                cur_p = cl_1[-1]
                if cur_p >= poc and poc not in support_levels_list:
                    support_levels_list.append(poc)
                    support_levels_list.sort()
                elif cur_p < poc and poc not in resistance_levels_list:
                    resistance_levels_list.append(poc)
                    resistance_levels_list.sort()
                if abs(cur_p - poc) <= (atr_1m * 1.5):
                    at_key_level = True

        # HTF Breakout & Retest Detection
        breakout_direction = "NONE"
        breakout_level = 0.0
        breakout_detected = False
        retest_detected = False

        if len(cl_1) >= 2:
            cur_c = cl_1[-1]
            prev_c = cl_1[-2]
            cur_vol = vol_1[-1] if len(vol_1) > 0 else 0.0
            avg_vol_20 = float(np.mean(vol_1[-20:])) if len(vol_1) >= 20 else (float(np.mean(vol_1)) if len(vol_1) > 0 else 1.0)
            rel_vol_1m = (cur_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

            # Bullish Breakout above PDH
            if pdh > 0 and cur_c > pdh and (prev_c <= pdh or (len(cl_1) >= 3 and cl_1[-3] <= pdh)):
                if (rel_vol_1m >= (self.volume_expansion_mult * 0.9) or is_squeeze or squeeze_fired) and displacement_1m:
                    breakout_detected = True
                    breakout_direction = "BULLISH"
                    breakout_level = pdh
            # Bearish Breakout below PDL
            elif pdl > 0 and cur_c < pdl and (prev_c >= pdl or (len(cl_1) >= 3 and cl_1[-3] >= pdl)):
                if (rel_vol_1m >= (self.volume_expansion_mult * 0.9) or is_squeeze or squeeze_fired) and displacement_1m:
                    breakout_detected = True
                    breakout_direction = "BEARISH"
                    breakout_level = pdl

            # Breakout Retest of broken level
            if not breakout_detected and len(cl_1) >= 5:
                # Bullish Retest of PDH
                if pdh > 0 and np.max(hi_1[-10:]) > pdh:
                    if abs(cur_c - pdh) <= (atr_1m * 0.8) or (lo_1[-1] <= pdh <= cur_c):
                        if cur_c >= op_1[-1]:  # Bullish reaction
                            retest_detected = True
                            breakout_direction = "BULLISH"
                            breakout_level = pdh
                # Bearish Retest of PDL
                elif pdl > 0 and np.min(lo_1[-10:]) < pdl:
                    if abs(cur_c - pdl) <= (atr_1m * 0.8) or (cur_c <= pdl <= hi_1[-1]):
                        if cur_c <= op_1[-1]:  # Bearish reaction
                            retest_detected = True
                            breakout_direction = "BEARISH"
                            breakout_level = pdl

        # Collect key levels dictionary for trap market detection
        key_levels_dict = {
            "PWH": pwh, "PWL": pwl,
            "PDH": pdh, "PDL": pdl,
            "EQH": eqh, "EQL": eql,
            "NEAREST_RESISTANCE": nearest_resistance,
            "NEAREST_SUPPORT": nearest_support,
            "VAH": vah, "VAL": val,
            "ASIAN_HIGH": asian_high, "ASIAN_LOW": asian_low,
        }

        # Timestamp of the latest candle
        latest_ts = None
        if isinstance(candles_1m, list) and len(candles_1m) > 0:
            last_c = candles_1m[-1]
            latest_ts = getattr(last_c, "open_time", None) if not isinstance(last_c, dict) else last_c.get("open_time")
        elif isinstance(candles_1m, dict) and "open_time" in candles_1m and len(candles_1m["open_time"]) > 0:
            latest_ts = candles_1m["open_time"][-1]

        # Run Trap Market Detection on 1m candles (fallback to 5m)
        is_bull_trap, is_bear_trap, is_judas_swing, is_vol_abs, trap_type_str, trap_lvl, trap_wick_ext = (
            detect_trap_market(
                op_1, hi_1, lo_1, cl_1, vol_1,
                key_levels=key_levels_dict,
                atr=atr_1m,
                wick_ratio_threshold=0.20,
                cur_timestamp_ms=latest_ts,
            )
        )
        if not (is_bull_trap or is_bear_trap or is_judas_swing or is_vol_abs) and len(cl_5) >= 2:
            is_bull_trap, is_bear_trap, is_judas_swing, is_vol_abs, trap_type_str, trap_lvl, trap_wick_ext = (
                detect_trap_market(
                    op_5, hi_5, lo_5, cl_5, vol_5,
                    key_levels=key_levels_dict,
                    atr=atr_1m * 2.0,
                    wick_ratio_threshold=0.25,
                    cur_timestamp_ms=latest_ts,
                )
            )

        # Determine Setup Type with high-probability ICT, Trap Market, and HTF Breakout confluence
        setup_type = "NONE"
        if breakout_detected:
            setup_type = "HTF_BREAKOUT"
            is_valid = True
            invalidation_reason = None
        elif retest_detected:
            setup_type = "BREAKOUT_RETEST"
            is_valid = True
            invalidation_reason = None
        elif (
            vah > 0 and len(cl_1) >= 2 and cl_1[-1] > vah and (cl_1[-2] <= vah or is_squeeze or squeeze_fired)
            and displacement_1m
        ):
            setup_type = "VALUE_AREA_BREAKOUT"
            breakout_direction = "BULLISH"
            breakout_level = vah
            is_valid = True
            invalidation_reason = None
        elif (
            val > 0 and len(cl_1) >= 2 and cl_1[-1] < val and (cl_1[-2] >= val or is_squeeze or squeeze_fired)
            and displacement_1m
        ):
            setup_type = "VALUE_AREA_BREAKOUT"
            breakout_direction = "BEARISH"
            breakout_level = val
            is_valid = True
            invalidation_reason = None
        elif liquidity_sweep_5m and (
            (eqh > 0 and sweep_direction == 'BEARISH') or
            (eql > 0 and sweep_direction == 'BULLISH') or
            (asian_high > 0 and sweep_direction == 'BEARISH') or
            (asian_low > 0 and sweep_direction == 'BULLISH')
        ):
            # Liquidity hunt on engineered liquidity pools (EQH/EQL or Asian H/L)
            setup_type = "LIQUIDITY_HUNT_REVERSAL"
            is_valid = True
            invalidation_reason = None
        elif ob_detected and ob_testing and not retest_1m and (
            (ob_direction == "BULLISH" and bias_15m == "BULLISH") or
            (ob_direction == "BEARISH" and bias_15m == "BEARISH")
        ):
            setup_type = "ORDER_BLOCK_PULLBACK"
            is_valid = True
            invalidation_reason = None
        elif liquidity_sweep_5m and fvg_detected and sweep_direction == fvg_dir:
            # Swept liquidity, displaced into an FVG, and returning: Ultimate ICT Setup
            setup_type = "SWEEP_AND_FVG"
            is_valid = True
            invalidation_reason = None
        elif liquidity_sweep_5m and sweep_direction in ('BULLISH', 'BEARISH'):
            setup_type = "SWEEP_REVERSAL"
            # A sweep of support/resistance is an intentional reversal scalp; mark valid
            is_valid = True
            invalidation_reason = None
        elif is_judas_swing:
            setup_type = "JUDAS_SWING_FADE"
            sweep_direction = "BEARISH" if is_bull_trap else "BULLISH"
            is_valid = True
            invalidation_reason = None
        elif is_bull_trap:
            setup_type = "BULL_TRAP_REVERSAL"
            sweep_direction = "BEARISH"
            is_valid = True
            invalidation_reason = None
        elif is_bear_trap:
            setup_type = "BEAR_TRAP_REVERSAL"
            sweep_direction = "BULLISH"
            is_valid = True
            invalidation_reason = None
        elif is_vol_abs:
            setup_type = "VOLUME_ABSORPTION_REVERSAL"
            sweep_direction = "BEARISH" if is_bull_trap else "BULLISH"
            is_valid = True
            invalidation_reason = None
        elif fvg_detected and fvg_testing and (
            (fvg_dir == "BULLISH" and bias_15m == "BULLISH") or
            (fvg_dir == "BEARISH" and bias_15m == "BEARISH")
        ):
            setup_type = "FVG_RETEST"
            is_valid = True
            invalidation_reason = None
        elif is_valid and (retest_1m or displacement_1m):
            setup_type = "TREND_PULLBACK"
        elif is_valid:
            setup_type = "TREND_CONTINUATION"

        # TradingView Chart Pattern & Premium/Discount Analysis
        sw_hi_5, sw_lo_5 = self.find_swing_points(hi_5, lo_5) if len(hi_5) >= 5 else ([], [])
        range_h = max(pdh, float(np.max(hi_15))) if len(hi_15) > 0 else (pdh if pdh > 0 else 0.0)
        range_l = min(pdl, float(np.min(lo_15))) if len(lo_15) > 0 and pdl > 0 else (pdl if pdl > 0 else 0.0)

        chart_res = ChartPatternEngine.analyze(
            high=hi_5,
            low=lo_5,
            close=cl_5,
            volume=vol_5,
            swing_highs=sw_hi_5,
            swing_lows=sw_lo_5,
            range_high=range_h,
            range_low=range_l,
        )

        # Classify Pro Trading Playbook for Jev AI
        if is_judas_swing or (liquidity_sweep_5m and (asian_high > 0 or asian_low > 0)) or is_bull_trap or is_bear_trap:
            playbook = "ICT_JUDAS_SWEEP"
        elif ob_testing or fvg_testing:
            playbook = "ORDER_BLOCK_FVG_PULLBACK"
        elif squeeze_fired or is_squeeze:
            playbook = "CARTER_SQUEEZE_BREAKOUT"
        elif at_key_level and (poc > 0):
            playbook = "VALUE_AREA_MEAN_REVERSION"
        elif chart_res.pattern_type != "NONE":
            playbook = f"CHART_{chart_res.pattern_type}"
        else:
            playbook = "TREND_CONTINUATION" if bias_15m != "NEUTRAL" else "RANGE_EXPANSION"

        return StructureAnalysis(
            bias_15m=bias_15m,
            bos_5m=bos_5m,
            choch_5m=choch_5m,
            bos_direction_5m=bos_direction_5m,
            liquidity_sweep_5m=liquidity_sweep_5m,
            displacement_1m=displacement_1m,
            retest_1m=retest_1m,
            is_valid=is_valid,
            invalidation_reason=invalidation_reason,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            support_levels=support_levels_list,
            resistance_levels=resistance_levels_list,
            setup_type=setup_type,
            sweep_direction=sweep_direction,
            at_key_level=at_key_level,
            fvg_detected=fvg_detected,
            fvg_direction=fvg_dir,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            fvg_testing=fvg_testing,
            pdh=pdh,
            pdl=pdl,
            poc=poc,
            vah=vah,
            val=val,
            eqh=eqh,
            eql=eql,
            asian_high=asian_high,
            asian_low=asian_low,
            ob_detected=ob_detected,
            ob_direction=ob_direction,
            ob_top=ob_top,
            ob_bottom=ob_bottom,
            ob_testing=ob_testing,
            is_squeeze=is_squeeze,
            squeeze_fired=squeeze_fired,
            breakout_level=breakout_level,
            breakout_direction=breakout_direction,
            pwh=pwh,
            pwl=pwl,
            is_bull_trap=is_bull_trap,
            is_bear_trap=is_bear_trap,
            is_judas_swing=is_judas_swing,
            is_volume_absorption=is_vol_abs,
            trap_level=trap_lvl,
            trap_wick_extreme=trap_wick_ext,
            chart_pattern=chart_res.pattern_type,
            chart_pattern_direction=chart_res.direction,
            chart_pattern_conf=chart_res.confidence,
            chart_pattern_level=chart_res.key_level,
            chart_pattern_target=chart_res.target_price,
            pricing_zone=chart_res.pricing_zone,
            range_position_pct=chart_res.range_position_pct,
            playbook=playbook,
        )
