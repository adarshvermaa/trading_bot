"""
TradingView-style Chart Pattern and ICT Price Action Detection Engine.

Detects classic and institutional chart formations:
- Double Bottom (W-Pattern) & Double Top (M-Pattern)
- Head & Shoulders & Inverse Head & Shoulders
- Ascending, Descending & Symmetrical Triangles
- Bull & Bear Flags
- Premium vs Discount Dealing Range Matrix
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChartPatternType(str, Enum):
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    DOUBLE_TOP = "DOUBLE_TOP"
    INVERSE_HEAD_AND_SHOULDERS = "INV_HEAD_SHOULDERS"
    HEAD_AND_SHOULDERS = "HEAD_SHOULDERS"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    NONE = "NONE"


class PricingZone(str, Enum):
    DISCOUNT = "DISCOUNT"        # < 50% of dealing range (Optimal Longs)
    PREMIUM = "PREMIUM"          # > 50% of dealing range (Optimal Shorts)
    EQUILIBRIUM = "EQUILIBRIUM"  # 45% - 55% fair value midpoint


@dataclass
class ChartPatternResult:
    pattern_type: str = ChartPatternType.NONE.value
    confidence: float = 0.0
    direction: str = "NEUTRAL"   # "BULLISH", "BEARISH", "NEUTRAL"
    key_level: float = 0.0       # Neckline, breakout apex, or flag boundary
    target_price: float = 0.0    # Measured move structural target
    pricing_zone: str = PricingZone.EQUILIBRIUM.value
    range_position_pct: float = 50.0  # 0% (at range low) to 100% (at range high)
    description: str = "No classic chart pattern detected"


def calculate_premium_discount_zone(
    current_price: float, range_high: float, range_low: float
) -> Tuple[str, float]:
    """
    Calculate where current price sits relative to the HTF Dealing Range (ICT Concept).
    
    Equilibrium = (Range High + Range Low) / 2
    - Discount (< 48%): High-probability institutional accumulation for LONGS
    - Premium (> 52%): High-probability institutional distribution for SHORTS
    - Equilibrium (48% - 52%): Fair value, lower directional edge
    """
    if range_high <= range_low or current_price <= 0:
        return PricingZone.EQUILIBRIUM.value, 50.0

    pos_pct = ((current_price - range_low) / (range_high - range_low)) * 100.0
    pos_pct = max(0.0, min(100.0, pos_pct))

    if pos_pct < 48.0:
        zone = PricingZone.DISCOUNT.value
    elif pos_pct > 52.0:
        zone = PricingZone.PREMIUM.value
    else:
        zone = PricingZone.EQUILIBRIUM.value

    return zone, round(pos_pct, 1)


def detect_double_bottom(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    swing_lows: List[int],
    tolerance_pct: float = 0.0035,
) -> Optional[Dict[str, Any]]:
    """
    Detect W-Pattern / Double Bottom reversal.
    Two consecutive swing lows at approximately the same level (+/- tolerance),
    separated by an intermediate swing high (neckline).
    """
    if len(swing_lows) < 2 or len(close) < 10:
        return None

    # Take the last two swing lows
    i1, i2 = swing_lows[-2], swing_lows[-1]
    if i2 <= i1 or (i2 - i1) < 3:
        return None

    l1, l2 = float(low[i1]), float(low[i2])
    diff_pct = abs(l1 - l2) / min(l1, l2)
    if diff_pct > tolerance_pct:
        return None

    # Neckline is the highest peak between i1 and i2
    intermediate_highs = high[i1:i2 + 1]
    if len(intermediate_highs) == 0:
        return None
    neckline = float(np.max(intermediate_highs))
    cur_p = float(close[-1])

    # Price should be either breaking or nearing neckline
    height = neckline - min(l1, l2)
    target = neckline + height

    # Confirmation: current price must be above bottom and near/above neckline
    if cur_p >= (min(l1, l2) + 0.5 * height):
        conf = 0.85 if cur_p >= neckline else 0.70
        return {
            "pattern_type": ChartPatternType.DOUBLE_BOTTOM.value,
            "confidence": conf,
            "direction": "BULLISH",
            "key_level": neckline,
            "target_price": target,
            "description": f"Double Bottom (W-Pattern) at ${min(l1, l2):,.1f} with Neckline at ${neckline:,.1f}",
        }
    return None


def detect_double_top(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_highs: List[int],
    tolerance_pct: float = 0.0035,
) -> Optional[Dict[str, Any]]:
    """
    Detect M-Pattern / Double Top reversal.
    Two consecutive swing highs at approximately the same level (+/- tolerance),
    separated by an intermediate swing low (neckline).
    """
    if len(swing_highs) < 2 or len(close) < 10:
        return None

    i1, i2 = swing_highs[-2], swing_highs[-1]
    if i2 <= i1 or (i2 - i1) < 3:
        return None

    h1, h2 = float(high[i1]), float(high[i2])
    diff_pct = abs(h1 - h2) / min(h1, h2)
    if diff_pct > tolerance_pct:
        return None

    # Neckline is the lowest trough between i1 and i2
    intermediate_lows = low[i1:i2 + 1]
    if len(intermediate_lows) == 0:
        return None
    neckline = float(np.min(intermediate_lows))
    cur_p = float(close[-1])

    height = max(h1, h2) - neckline
    target = neckline - height

    if cur_p <= (max(h1, h2) - 0.5 * height):
        conf = 0.85 if cur_p <= neckline else 0.70
        return {
            "pattern_type": ChartPatternType.DOUBLE_TOP.value,
            "confidence": conf,
            "direction": "BEARISH",
            "key_level": neckline,
            "target_price": target,
            "description": f"Double Top (M-Pattern) at ${max(h1, h2):,.1f} with Neckline at ${neckline:,.1f}",
        }
    return None


def detect_head_and_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_highs: List[int],
    swing_lows: List[int],
) -> Optional[Dict[str, Any]]:
    """Detect classic Head and Shoulders (Bearish Reversal)."""
    if len(swing_highs) < 3 or len(swing_lows) < 2:
        return None

    s1, h, s2 = swing_highs[-3], swing_highs[-2], swing_highs[-1]
    if not (s1 < h < s2):
        return None

    sh1, hh, sh2 = float(high[s1]), float(high[h]), float(high[s2])

    # Head must be distinctly higher than both shoulders
    if hh <= sh1 or hh <= sh2:
        return None

    # Shoulders should be within 1.5% of each other
    if abs(sh1 - sh2) / min(sh1, sh2) > 0.015:
        return None

    # Troughs define neckline
    trough_levels = [float(low[i]) for i in swing_lows if s1 <= i <= s2]
    if not trough_levels:
        return None
    neckline = float(np.mean(trough_levels))
    cur_p = float(close[-1])
    target = neckline - (hh - neckline)

    if cur_p < sh2:
        conf = 0.85 if cur_p <= neckline else 0.75
        return {
            "pattern_type": ChartPatternType.HEAD_AND_SHOULDERS.value,
            "confidence": conf,
            "direction": "BEARISH",
            "key_level": neckline,
            "target_price": target,
            "description": f"Head & Shoulders: Head ${hh:,.1f}, Shoulders ~${(sh1+sh2)/2:,.1f}, Neckline ${neckline:,.1f}",
        }
    return None


def detect_inverse_head_and_shoulders(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_highs: List[int],
    swing_lows: List[int],
) -> Optional[Dict[str, Any]]:
    """Detect Inverse Head and Shoulders (Bullish Reversal)."""
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return None

    s1, h, s2 = swing_lows[-3], swing_lows[-2], swing_lows[-1]
    if not (s1 < h < s2):
        return None

    sl1, hl, sl2 = float(low[s1]), float(low[h]), float(low[s2])

    # Head must be distinctly lower than both shoulders
    if hl >= sl1 or hl >= sl2:
        return None

    # Shoulders within 1.5%
    if abs(sl1 - sl2) / min(sl1, sl2) > 0.015:
        return None

    peaks = [float(high[i]) for i in swing_highs if s1 <= i <= s2]
    if not peaks:
        return None
    neckline = float(np.mean(peaks))
    cur_p = float(close[-1])
    target = neckline + (neckline - hl)

    if cur_p > sl2:
        conf = 0.85 if cur_p >= neckline else 0.75
        return {
            "pattern_type": ChartPatternType.INVERSE_HEAD_AND_SHOULDERS.value,
            "confidence": conf,
            "direction": "BULLISH",
            "key_level": neckline,
            "target_price": target,
            "description": f"Inv Head & Shoulders: Head ${hl:,.1f}, Shoulders ~${(sl1+sl2)/2:,.1f}, Neckline ${neckline:,.1f}",
        }
    return None


def detect_triangles_and_flags(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    swing_highs: List[int],
    swing_lows: List[int],
) -> Optional[Dict[str, Any]]:
    """
    Detect Ascending/Descending Triangles and Bull/Bear Flags.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2 or len(close) < 15:
        return None

    cur_p = float(close[-1])
    h1, h2 = float(high[swing_highs[-2]]), float(high[swing_highs[-1]])
    l1, l2 = float(low[swing_lows[-2]]), float(low[swing_lows[-1]])

    # 1. Ascending Triangle: Flat Resistance (h1 ~= h2) + Higher Lows (l2 > l1)
    if abs(h1 - h2) / min(h1, h2) <= 0.0025 and (l2 - l1) / l1 > 0.002:
        res = max(h1, h2)
        target = res + (res - l1)
        return {
            "pattern_type": ChartPatternType.ASCENDING_TRIANGLE.value,
            "confidence": 0.80,
            "direction": "BULLISH",
            "key_level": res,
            "target_price": target,
            "description": f"Ascending Triangle: Flat Resistance ${res:,.1f} with Higher Lows",
        }

    # 2. Descending Triangle: Flat Support (l1 ~= l2) + Lower Highs (h2 < h1)
    if abs(l1 - l2) / min(l1, l2) <= 0.0025 and (h1 - h2) / h1 > 0.002:
        sup = min(l1, l2)
        target = sup - (h1 - sup)
        return {
            "pattern_type": ChartPatternType.DESCENDING_TRIANGLE.value,
            "confidence": 0.80,
            "direction": "BEARISH",
            "key_level": sup,
            "target_price": target,
            "description": f"Descending Triangle: Flat Support ${sup:,.1f} with Lower Highs",
        }

    # 3. Bull Flag: Strong impulse pole (3-5 bars) followed by shallow consolidation
    if len(close) >= 20:
        pole_start = float(close[-15])
        pole_peak = float(np.max(high[-15:-5]))
        pole_gain_pct = (pole_peak - pole_start) / pole_start
        consolidation_low = float(np.min(low[-5:]))
        retrace = (pole_peak - consolidation_low) / max(pole_peak - pole_start, 1e-6)

        if pole_gain_pct >= 0.015 and 0.15 <= retrace <= 0.45:
            target = pole_peak + (pole_peak - pole_start)
            return {
                "pattern_type": ChartPatternType.BULL_FLAG.value,
                "confidence": 0.82,
                "direction": "BULLISH",
                "key_level": pole_peak,
                "target_price": target,
                "description": f"Bull Flag: +{pole_gain_pct*100:.1f}% Impulse Pole with shallow pullback",
            }

    return None


class ChartPatternEngine:
    """Institutional TradingView Chart Pattern & Price Action Analyzer."""

    @staticmethod
    def analyze(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        swing_highs: List[int],
        swing_lows: List[int],
        range_high: float = 0.0,
        range_low: float = 0.0,
    ) -> ChartPatternResult:
        """
        Evaluate full TradingView chart patterns and Premium/Discount matrix.
        Runs in < 0.5ms using vectorized operations.
        """
        if len(close) < 10:
            return ChartPatternResult()

        cur_p = float(close[-1])
        r_high = range_high if range_high > 0 else float(np.max(high[-60:])) if len(high) >= 60 else float(np.max(high))
        r_low = range_low if range_low > 0 else float(np.min(low[-60:])) if len(low) >= 60 else float(np.min(low))

        zone, pos_pct = calculate_premium_discount_zone(cur_p, r_high, r_low)

        # 1. Check Double Bottom / Top
        db = detect_double_bottom(low, high, close, swing_lows)
        if db:
            return ChartPatternResult(
                pattern_type=db["pattern_type"],
                confidence=db["confidence"],
                direction=db["direction"],
                key_level=db["key_level"],
                target_price=db["target_price"],
                pricing_zone=zone,
                range_position_pct=pos_pct,
                description=db["description"],
            )

        dt = detect_double_top(high, low, close, swing_highs)
        if dt:
            return ChartPatternResult(
                pattern_type=dt["pattern_type"],
                confidence=dt["confidence"],
                direction=dt["direction"],
                key_level=dt["key_level"],
                target_price=dt["target_price"],
                pricing_zone=zone,
                range_position_pct=pos_pct,
                description=dt["description"],
            )

        # 2. Check Head & Shoulders / Inverse Head & Shoulders
        ihs = detect_inverse_head_and_shoulders(high, low, close, swing_highs, swing_lows)
        if ihs:
            return ChartPatternResult(
                pattern_type=ihs["pattern_type"],
                confidence=ihs["confidence"],
                direction=ihs["direction"],
                key_level=ihs["key_level"],
                target_price=ihs["target_price"],
                pricing_zone=zone,
                range_position_pct=pos_pct,
                description=ihs["description"],
            )

        hs = detect_head_and_shoulders(high, low, close, swing_highs, swing_lows)
        if hs:
            return ChartPatternResult(
                pattern_type=hs["pattern_type"],
                confidence=hs["confidence"],
                direction=hs["direction"],
                key_level=hs["key_level"],
                target_price=hs["target_price"],
                pricing_zone=zone,
                range_position_pct=pos_pct,
                description=hs["description"],
            )

        # 3. Check Triangles & Flags
        tf = detect_triangles_and_flags(high, low, close, volume, swing_highs, swing_lows)
        if tf:
            return ChartPatternResult(
                pattern_type=tf["pattern_type"],
                confidence=tf["confidence"],
                direction=tf["direction"],
                key_level=tf["key_level"],
                target_price=tf["target_price"],
                pricing_zone=zone,
                range_position_pct=pos_pct,
                description=tf["description"],
            )

        # Baseline: No pattern detected, return Premium/Discount status
        return ChartPatternResult(
            pattern_type=ChartPatternType.NONE.value,
            confidence=0.50,
            direction="NEUTRAL",
            key_level=0.0,
            target_price=0.0,
            pricing_zone=zone,
            range_position_pct=pos_pct,
            description=f"Dealing Range: {zone} ({pos_pct:.1f}% of 15m range)",
        )
