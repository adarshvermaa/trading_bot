import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from src.utils.logger import get_logger

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

    def __post_init__(self):
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []

class MarketStructure:
    def __init__(
        self, 
        config: Optional[Any] = None,
        lookback: int = 10, 
        wick_ratio_threshold: float = 0.6, 
        displacement_threshold: float = 0.7, 
        retest_tolerance_mult: float = 0.5
    ):
        if config is not None and not isinstance(config, int):
            self.lookback = getattr(config, "min_swing_lookback", getattr(config, "lookback", 10))
            self.wick_ratio_threshold = getattr(config, "liquidity_sweep_wick_ratio", getattr(config, "wick_ratio_threshold", 0.6))
            self.displacement_threshold = getattr(config, "displacement_body_ratio", getattr(config, "displacement_threshold", 0.7))
            self.retest_tolerance_mult = getattr(config, "retest_tolerance_atr_mult", getattr(config, "retest_tolerance_mult", 0.5))
        else:
            self.lookback = config if isinstance(config, int) else lookback
            self.wick_ratio_threshold = wick_ratio_threshold
            self.displacement_threshold = displacement_threshold
            self.retest_tolerance_mult = retest_tolerance_mult

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

    def analyze(self, candles_15m: List[Any], candles_5m: List[Any], candles_1m: List[Any], atr_1m: float) -> StructureAnalysis:
        logger.info("Analyzing market structure across timeframes")
        
        # 15m analysis
        op_15, hi_15, lo_15, cl_15, vol_15 = self._extract_arrays(candles_15m)
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
        
        if len(hi_5) > 2 * self.lookback:
            sh_5, sl_5 = self.find_swing_points(hi_5, lo_5)
            
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

        # 1m analysis
        op_1, hi_1, lo_1, cl_1, vol_1 = self._extract_arrays(candles_1m)
        displacement_1m = False
        retest_1m = False
        
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

        # Also from 5m liquidity sweep:
        if sweep_direction == 'NONE' and liquidity_sweep_5m:
            if sh_5 and hi_5[-1] > hi_5[sh_5[-1]]:
                sweep_direction = 'BEARISH'
            elif sl_5 and lo_5[-1] < lo_5[sl_5[-1]]:
                sweep_direction = 'BULLISH'

        # Determine Setup Type
        setup_type = "NONE"
        if liquidity_sweep_5m and sweep_direction in ('BULLISH', 'BEARISH'):
            setup_type = "SWEEP_REVERSAL"
            # A sweep of support/resistance is an intentional reversal scalp; mark valid
            is_valid = True
            invalidation_reason = None
        elif is_valid and (retest_1m or displacement_1m):
            setup_type = "TREND_PULLBACK"
        elif is_valid:
            setup_type = "TREND_CONTINUATION"

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
        )
