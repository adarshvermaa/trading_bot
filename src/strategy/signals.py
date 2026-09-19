import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Any
from src.strategy.structure import StructureAnalysis

@dataclass
class Signal:
    direction: str
    strength: float
    entry_price: float
    sl_price: float
    tp_price: float
    ema_cross: str
    vwap_position: str
    rsi: float
    atr: float
    relative_volume: float
    adx: float
    timeframe_alignment: bool
    setup_type: str = "NONE"
    pattern: str = "NONE"

class SignalGenerator:
    def __init__(self, config: Any = None):
        self.config = config

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros_like(prices)
        if len(prices) == 0:
            return ema
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema

    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        typical_price = (high + low + close) / 3
        cum_vol = np.cumsum(volume)
        cum_pv = np.cumsum(typical_price * volume)
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = np.where(cum_vol == 0, typical_price, cum_pv / cum_vol)
        return vwap

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        rsi = np.zeros_like(prices)
        if len(prices) <= period:
            return rsi
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0:
            rs = 100
        else:
            rs = up / down
        rsi[period] = 100. - 100. / (1. + rs)
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi[i] = 100.
            else:
                rs = up / down
                rsi[i] = 100. - 100. / (1. + rs)
        return rsi

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        atr = np.zeros_like(close)
        if len(close) == 0:
            return atr
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
        atr[0] = tr[0]
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    @staticmethod
    def calculate_relative_volume(volume: np.ndarray, period: int = 20) -> np.ndarray:
        rel_vol = np.zeros_like(volume)
        if len(volume) < period:
            return rel_vol
        sma = np.convolve(volume, np.ones(period)/period, mode='valid')
        safe_sma = np.where(sma > 0, sma, np.nan)
        rel_vol[period-1:] = np.nan_to_num(volume[period-1:] / safe_sma, nan=1.0, posinf=1.0, neginf=1.0)
        return rel_vol

    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        adx = np.zeros_like(close)
        if len(close) <= period:
            return adx
        
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
                
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
        smooth_plus_dm = np.zeros_like(close)
        smooth_minus_dm = np.zeros_like(close)
        smooth_tr = np.zeros_like(close)
        
        smooth_plus_dm[period] = np.sum(plus_dm[1:period+1])
        smooth_minus_dm[period] = np.sum(minus_dm[1:period+1])
        smooth_tr[period] = np.sum(tr[1:period+1])
        
        for i in range(period + 1, len(close)):
            smooth_plus_dm[i] = smooth_plus_dm[i-1] - (smooth_plus_dm[i-1] / period) + plus_dm[i]
            smooth_minus_dm[i] = smooth_minus_dm[i-1] - (smooth_minus_dm[i-1] / period) + minus_dm[i]
            smooth_tr[i] = smooth_tr[i-1] - (smooth_tr[i-1] / period) + tr[i]
            
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if smooth_tr[i] != 0:
                plus_di[i] = 100 * (smooth_plus_dm[i] / smooth_tr[i])
                minus_di[i] = 100 * (smooth_minus_dm[i] / smooth_tr[i])
                
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                
        adx[period] = np.mean(dx[period:period*2]) if len(dx) >= period*2 else np.mean(dx[period:])
        for i in range(period + 1, len(close)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
            
        return adx

    @staticmethod
    def detect_candlestick_pattern(
        op: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray
    ) -> str:
        """Detect key 1m trigger candlestick patterns (Pinbar / Hammer, Shooting Star, Engulfing)."""
        if len(cl) < 2:
            return "NONE"

        o_curr, h_curr, l_curr, c_curr = op[-1], hi[-1], lo[-1], cl[-1]
        rng_curr = h_curr - l_curr
        if rng_curr <= 0:
            return "NONE"

        body_curr = abs(c_curr - o_curr)
        lower_wick = min(o_curr, c_curr) - l_curr
        upper_wick = h_curr - max(o_curr, c_curr)

        # 1. Bullish Pinbar / Hammer (lower wick >= 50% range, small upper wick <= 25%)
        if (lower_wick / rng_curr) >= 0.50 and (upper_wick / rng_curr) <= 0.25:
            return "HAMMER"

        # 2. Bearish Pinbar / Shooting Star (upper wick >= 50% range, small lower wick <= 25%)
        if (upper_wick / rng_curr) >= 0.50 and (lower_wick / rng_curr) <= 0.25:
            return "SHOOTING_STAR"

        # 3. Engulfing patterns
        o_prev, c_prev = op[-2], cl[-2]
        body_prev = abs(c_prev - o_prev)
        prev_bearish = c_prev < o_prev
        prev_bullish = c_prev > o_prev
        curr_bullish = c_curr > o_curr
        curr_bearish = c_curr < o_curr

        if prev_bearish and curr_bullish and c_curr >= o_prev and o_curr <= c_prev and body_curr > body_prev:
            return "BULLISH_ENGULFING"

        if prev_bullish and curr_bearish and c_curr <= o_prev and o_curr >= c_prev and body_curr > body_prev:
            return "BEARISH_ENGULFING"

        return "NONE"

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
            return (np.array([c['open'] for c in candles]),
                    np.array([c['high'] for c in candles]),
                    np.array([c['low'] for c in candles]),
                    np.array([c['close'] for c in candles]),
                    np.array([c['volume'] for c in candles]))
        return (np.array([c.open for c in candles]),
                np.array([c.high for c in candles]),
                np.array([c.low for c in candles]),
                np.array([c.close for c in candles]),
                np.array([c.volume for c in candles]))

    def generate(self, candles_15m: List[Any], candles_5m: List[Any], candles_1m: List[Any], structure: StructureAnalysis) -> Signal:
        op, hi, lo, cl, vol = self._extract_arrays(candles_1m)
        if len(cl) == 0:
            return Signal('NONE', 0.0, 0.0, 0.0, 0.0, 'NEUTRAL', 'ABOVE', 50.0, 0.0, 1.0, 0.0, False, "NONE", "NONE")
            
        ema_20 = self.calculate_ema(cl, 20)
        ema_50 = self.calculate_ema(cl, 50)
        vwap = self.calculate_vwap(hi, lo, cl, vol)
        rsi = self.calculate_rsi(cl, 14)
        atr = self.calculate_atr(hi, lo, cl, 14)
        rel_vol = self.calculate_relative_volume(vol, 20)
        adx = self.calculate_adx(hi, lo, cl, 14)
        
        last_close = cl[-1]
        cur_atr = float(atr[-1]) if len(atr) > 0 else 0.0
        
        ema_cross = 'NEUTRAL'
        if ema_20[-1] > ema_50[-1]:
            ema_cross = 'BULLISH'
        elif ema_20[-1] < ema_50[-1]:
            ema_cross = 'BEARISH'
            
        vwap_pos = 'ABOVE' if last_close > vwap[-1] else 'BELOW'
        pattern = self.detect_candlestick_pattern(op, hi, lo, cl)

        direction = 'NONE'
        strength = 0.0
        setup_type = getattr(structure, "setup_type", "NONE")

        # --- Dual High-Accuracy Scalp Setups ---
        if setup_type == "SWEEP_REVERSAL":
            sweep_dir = getattr(structure, "sweep_direction", "NONE")
            if sweep_dir == "BULLISH":
                direction = "LONG"
                strength = 0.88
                if pattern in ("HAMMER", "BULLISH_ENGULFING"):
                    strength = min(0.95, strength + 0.05)
            elif sweep_dir == "BEARISH":
                direction = "SHORT"
                strength = 0.88
                if pattern in ("SHOOTING_STAR", "BEARISH_ENGULFING"):
                    strength = min(0.95, strength + 0.05)
        elif setup_type == "TREND_PULLBACK":
            if structure.bias_15m == "BULLISH":
                direction = "LONG"
                val_high = max(ema_20[-1], vwap[-1])
                val_low = min(ema_50[-1], vwap[-1]) - (cur_atr * 0.5)
                if lo[-1] <= val_high and cl[-1] >= val_low:
                    strength = 0.85
                else:
                    strength = 0.80
                if pattern in ("HAMMER", "BULLISH_ENGULFING"):
                    strength = min(0.95, strength + 0.05)
            elif structure.bias_15m == "BEARISH":
                direction = "SHORT"
                val_low = min(ema_20[-1], vwap[-1])
                val_high = max(ema_50[-1], vwap[-1]) + (cur_atr * 0.5)
                if hi[-1] >= val_low and cl[-1] <= val_high:
                    strength = 0.85
                else:
                    strength = 0.80
                if pattern in ("SHOOTING_STAR", "BEARISH_ENGULFING"):
                    strength = min(0.95, strength + 0.05)
        elif structure.is_valid:
            if structure.bias_15m == 'BULLISH':
                direction = 'LONG'
                strength = 0.80 if structure.is_valid else 0.0
            elif structure.bias_15m == 'BEARISH':
                direction = 'SHORT'
                strength = 0.80 if structure.is_valid else 0.0

        timeframe_alignment = (
            (structure.bias_15m == ema_cross)
            or (direction == "LONG" and ema_cross == "BULLISH")
            or (direction == "SHORT" and ema_cross == "BEARISH")
            or (setup_type == "SWEEP_REVERSAL" and direction in ("LONG", "SHORT"))
        )

        return Signal(
            direction=direction,
            strength=strength if (structure.is_valid or setup_type == "SWEEP_REVERSAL") else 0.0,
            entry_price=last_close,
            sl_price=last_close * 0.99 if direction == 'LONG' else last_close * 1.01,
            tp_price=last_close * 1.02 if direction == 'LONG' else last_close * 0.98,
            ema_cross=ema_cross,
            vwap_position=vwap_pos,
            rsi=float(rsi[-1]),
            atr=cur_atr,
            relative_volume=float(rel_vol[-1]),
            adx=float(adx[-1]),
            timeframe_alignment=timeframe_alignment,
            setup_type=setup_type,
            pattern=pattern,
        )


    def evaluate_position_health(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        candles_1m: Any,
        candles_5m: Any,
        nearest_support: float = 0.0,
        nearest_resistance: float = 0.0,
        position_age_seconds: float = 0.0,
    ) -> Tuple[str, str]:
        """Evaluate health of an active position based on live technical analysis.
        
        Returns: (health_status, reason)
            health_status: 'STRONG', 'WEAK', or 'EXIT'
            reason: human-readable explanation
        """
        op, hi, lo, cl, vol = self._extract_arrays(candles_1m)
        if len(cl) < 20:
            return 'STRONG', 'Insufficient data'
        
        ema_20 = self.calculate_ema(cl, 20)
        ema_50 = self.calculate_ema(cl, 50)
        rsi = self.calculate_rsi(cl, 14)
        current_rsi = float(rsi[-1])
        
        is_long = position_side.upper() in ('LONG', 'BUY')
        issues = []
        
        # 1. EMA cross against position
        if is_long and ema_20[-1] < ema_50[-1] and ema_20[-2] >= ema_50[-2]:
            issues.append('EMA bearish cross')
        elif not is_long and ema_20[-1] > ema_50[-1] and ema_20[-2] <= ema_50[-2]:
            issues.append('EMA bullish cross')
        
        # 2. RSI extreme against position
        if is_long and current_rsi > 80:
            issues.append(f'RSI overbought ({current_rsi:.1f})')
        elif not is_long and current_rsi < 20:
            issues.append(f'RSI oversold ({current_rsi:.1f})')
        
        # 3. Price broke below support (LONG) or above resistance (SHORT)
        if is_long and nearest_support > 0 and current_price < nearest_support:
            issues.append(f'Price below support ${nearest_support:,.2f}')
        elif not is_long and nearest_resistance > 0 and current_price > nearest_resistance:
            issues.append(f'Price above resistance ${nearest_resistance:,.2f}')
        
        # 4. Check 5m structure for BOS against position
        op5, hi5, lo5, cl5, vol5 = self._extract_arrays(candles_5m)
        if len(cl5) >= 3:
            if is_long:
                # If last 3 candles all closed lower (strong bearish momentum)
                if cl5[-1] < cl5[-2] < cl5[-3]:
                    issues.append('5m bearish momentum (3 consecutive lower closes)')
            else:
                if cl5[-1] > cl5[-2] > cl5[-3]:
                    issues.append('5m bullish momentum (3 consecutive higher closes)')
        
        # 5. Price below EMA50 for LONG or above EMA50 for SHORT
        if len(ema_50) > 0:
            if is_long and current_price < ema_50[-1]:
                issues.append('Price below EMA50')
            elif not is_long and current_price > ema_50[-1]:
                issues.append('Price above EMA50')
        
        # 6. Time-based stall protection for ultra-fast scalping
        if position_age_seconds > 300:  # > 5 minutes
            pct_move = abs(current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            if pct_move < 0.0003:  # less than 0.03% move in 5 minutes
                issues.append("Time-stop: Position stagnating > 5m without momentum")
        elif position_age_seconds > 180:  # > 3 minutes
            pct_move = abs(current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            if pct_move < 0.00015:  # flat position
                issues.append("Stall warning: Trade flat after 3m")
        
        # Determine health status
        if len(issues) >= 3:
            return 'EXIT', '; '.join(issues)
        elif len(issues) >= 1:
            return 'WEAK', '; '.join(issues)
        else:
            return 'STRONG', 'All indicators aligned'


# Top-level standalone / helper function aliases
calculate_atr = SignalGenerator.calculate_atr
compute_atr = SignalGenerator.calculate_atr
calculate_adx = SignalGenerator.calculate_adx
compute_adx = SignalGenerator.calculate_adx
calculate_ema = SignalGenerator.calculate_ema
compute_ema = SignalGenerator.calculate_ema
calculate_vwap = SignalGenerator.calculate_vwap
compute_vwap = SignalGenerator.calculate_vwap
calculate_rsi = SignalGenerator.calculate_rsi
compute_rsi = SignalGenerator.calculate_rsi
calculate_relative_volume = SignalGenerator.calculate_relative_volume
compute_relative_volume = SignalGenerator.calculate_relative_volume
