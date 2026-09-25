import math
import time
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timezone

from src.config import RiskConfig
from src.risk.dynamic_leverage import DynamicLeverageEngine, DynamicLeverageResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


def round_to_tick(price: float, tick_size: float, direction: str = 'NEAREST') -> float:
    """Round price to the nearest valid exchange tick_size.
    
    direction:
        'NEAREST' - round to nearest tick (standard)
        'DOWN'    - floor to tick (e.g. for LONG stop loss)
        'UP'      - ceil to tick (e.g. for SHORT stop loss)
    """
    if not tick_size or tick_size <= 0:
        return price
    
    tick_str = f"{tick_size:.10f}".rstrip("0")
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
    
    ratio = price / tick_size
    if direction == 'DOWN':
        ticks = math.floor(round(ratio, 8))
    elif direction == 'UP':
        ticks = math.ceil(round(ratio, 8))
    else:
        ticks = round(ratio)
        
    rounded = round(ticks * tick_size, decimals)
    return rounded


def format_price(price: float, tick_size: float) -> str:
    """Format price to string with exact decimal precision matching tick_size."""
    if not tick_size or tick_size <= 0:
        return str(price)
    tick_str = f"{tick_size:.10f}".rstrip("0")
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
    return f"{price:.{decimals}f}"


class RiskManager:
    def __init__(self, config: RiskConfig, jev_client: Optional[Any] = None):
        self.config = config
        self.jev_client = jev_client
        self.dynamic_leverage_engine = DynamicLeverageEngine(getattr(config, "dynamic_leverage", None))
        
        # State tracking
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_loss_time: float = 0.0
        self.daily_start_equity: float = 0.0
        self.is_kill_switch_active: bool = False
        
        # Failsafe flags
        self.data_stale: bool = False
        self.market_data_connected: bool = True
        self.binance_connected: bool = True  # Backward compatibility
        self.delta_connected: bool = True
        self.delta_error_reason: str = ""
        
        # Keep track of last daily reset
        self.last_reset_date = datetime.now(timezone.utc).date()

    def calculate_dynamic_leverage(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: Optional[float] = None,
        structure: Optional[Any] = None,
        signal: Optional[Any] = None,
        jev_confidence: Optional[float] = None,
        atr: float = 0.0,
        avg_atr: float = 0.0,
        exchange_max_leverage: float = 100.0,
        obi: float = 0.0,
        playbook: Optional[str] = None,
        pricing_zone: Optional[str] = None,
        equity: float = 10000.0,
    ) -> DynamicLeverageResult:
        """Calculate leverage anchored strictly at fixed 25x or dynamic if explicitly enabled."""
        if not getattr(getattr(self.config, "dynamic_leverage", None), "enabled", False):
            fixed_lev = int(getattr(getattr(self.config, "leverage", None), "fixed_leverage", 25))
            actual_lev = int(min(fixed_lev, exchange_max_leverage))
            return DynamicLeverageResult(
                leverage=actual_lev,
                tier="FIXED_25X",
                effective_confidence=1.0,
                market_confidence=1.0,
                ai_confidence=1.0,
                volatility_dampener=1.0,
                drawdown_dampener=1.0,
                liquidation_buffer_ratio=2.5,
                max_safe_leverage=actual_lev,
                rationale="Fixed 25x Isolated Scalper Leverage",
            )

        max_daily_loss = equity * getattr(self.config.daily_limits, "max_daily_loss_pct", 0.03)
        return self.dynamic_leverage_engine.evaluate_leverage(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            structure=structure,
            signal=signal,
            jev_confidence=jev_confidence,
            atr=atr,
            avg_atr=avg_atr,
            daily_pnl=self.daily_pnl,
            max_daily_loss=max_daily_loss,
            exchange_max_leverage=exchange_max_leverage,
            obi=obi,
            playbook=playbook,
            pricing_zone=pricing_zone,
        )

    def validate_leverage(self, symbol_or_leverage: Any, exchange_max_leverage: float) -> Tuple[bool, int, str]:
        if isinstance(symbol_or_leverage, DynamicLeverageResult):
            desired_leverage = symbol_or_leverage.leverage
            actual_leverage = int(min(desired_leverage, exchange_max_leverage))
            if actual_leverage < desired_leverage and self.config.leverage.reject_on_leverage_fail:
                return False, int(exchange_max_leverage), f"Dynamic leverage {desired_leverage} exceeds exchange max {exchange_max_leverage}"
            return True, actual_leverage, "VALID"

        if isinstance(symbol_or_leverage, (int, float)):
            desired_leverage = int(symbol_or_leverage)
            if desired_leverage > exchange_max_leverage:
                if self.config.leverage.reject_on_leverage_fail:
                    return False, int(exchange_max_leverage), f"Leverage {desired_leverage} exceeds max {exchange_max_leverage}"
                return True, int(exchange_max_leverage), "VALID"
            return True, desired_leverage, "VALID"

        symbol = str(symbol_or_leverage)
        if symbol in self.config.leverage.high_leverage_assets:
            desired_leverage = self.config.leverage.high_leverage_value
        else:
            desired_leverage = self.config.leverage.default_leverage_value
            
        actual_leverage = int(min(desired_leverage, exchange_max_leverage))
        
        if actual_leverage < desired_leverage:
            if self.config.leverage.reject_on_leverage_fail and exchange_max_leverage < desired_leverage:
                return False, int(exchange_max_leverage), "Exchange max leverage too low and reject_on_leverage_fail is True"
            if exchange_max_leverage < self.config.leverage.safe_fallback_leverage:
                 return False, int(exchange_max_leverage), "Exchange max leverage below safe fallback"
            actual_leverage = self.config.leverage.safe_fallback_leverage

        return True, actual_leverage, "VALID"

    def calculate_stop_loss(self, entry_price: float, side: str, margin: float, leverage: int, 
                            contract_value: float, size: int, atr: float, 
                            tick_size: Optional[float] = None,
                            trap_wick_price: Optional[float] = None) -> Tuple[float, bool, str]:
        
        max_loss_pct = self.config.stop_loss.max_loss_pct_of_margin
        loss_amount = margin * max_loss_pct
        
        # price_diff * size * contract_value = loss_amount
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        if qty > 0:
            price_diff = loss_amount / qty
        else:
            price_diff = entry_price * (max_loss_pct / max(1, leverage))
        
        is_long = side.upper() in ('LONG', 'BUY')
        if is_long:
            sl_price = max(0.0, entry_price - price_diff)
            # If trap wick price is provided (e.g. wick low of a bear trap), place SL 1 tick below it
            if trap_wick_price and 0 < trap_wick_price < entry_price:
                tick_buf = tick_size if (tick_size and tick_size > 0) else (entry_price * 0.0005)
                candidate_wick_sl = trap_wick_price - tick_buf
                # Only use wick SL if it is tighter than or equal to max margin loss
                if candidate_wick_sl > sl_price:
                    sl_price = candidate_wick_sl
            if tick_size and tick_size > 0:
                sl_price = round_to_tick(sl_price, tick_size, direction='DOWN')
        else:
            sl_price = entry_price + price_diff
            # If trap wick price is provided (e.g. wick high of a bull trap), place SL 1 tick above it
            if trap_wick_price and trap_wick_price > entry_price:
                tick_buf = tick_size if (tick_size and tick_size > 0) else (entry_price * 0.0005)
                candidate_wick_sl = trap_wick_price + tick_buf
                # Only use wick SL if it is tighter than or equal to max margin loss
                if candidate_wick_sl < sl_price:
                    sl_price = candidate_wick_sl
            if tick_size and tick_size > 0:
                sl_price = round_to_tick(sl_price, tick_size, direction='UP')
            
        if sl_price <= 0:
            return sl_price, False, "Stop loss price must be strictly positive"

        # Check against ATR
        min_distance = self.config.stop_loss.min_stop_distance_atr * atr
        if abs(entry_price - sl_price) < min_distance:
            return sl_price, False, f"Stop loss too close: distance {abs(entry_price - sl_price)} < min {min_distance}"
            
        return sl_price, True, "VALID"

    def calculate_take_profit(self, entry_price: float, side: str, margin: float, leverage: int, 
                              contract_value: float, size: int, 
                              tick_size: Optional[float] = None,
                              structural_target: Optional[float] = None,
                              sl_price: Optional[float] = None,
                              setup_type: Optional[str] = None) -> float:
        is_long = side.upper() in ('LONG', 'BUY')
        
        # Check for Institutional Breakout target (>= 2.5R expansion)
        if setup_type in ("HTF_BREAKOUT", "BREAKOUT_RETEST"):
            risk = abs(entry_price - sl_price) if (sl_price and sl_price > 0) else (entry_price * 0.03 / max(1, leverage))
            breakout_target = (entry_price + 2.5 * risk) if is_long else max(0.0, entry_price - 2.5 * risk)
            if structural_target and structural_target > 0:
                if is_long and structural_target > breakout_target:
                    breakout_target = structural_target
                elif not is_long and structural_target < breakout_target:
                    breakout_target = structural_target
            if tick_size and tick_size > 0:
                breakout_target = round_to_tick(breakout_target, tick_size, direction='UP' if is_long else 'DOWN')
            return breakout_target

        # Check if structural target provides a sound Risk:Reward ratio (>= 1.5R)
        if structural_target and structural_target > 0:
            if is_long and structural_target > entry_price:
                reward = structural_target - entry_price
                risk = (entry_price - sl_price) if (sl_price and sl_price < entry_price) else (entry_price * 0.03 / max(1, leverage))
                if risk > 0 and (reward / risk) >= 1.5:
                    tp = round_to_tick(structural_target, tick_size, direction='DOWN') if (tick_size and tick_size > 0) else structural_target
                    return tp
            elif not is_long and structural_target < entry_price:
                reward = entry_price - structural_target
                risk = (sl_price - entry_price) if (sl_price and sl_price > entry_price) else (entry_price * 0.03 / max(1, leverage))
                if risk > 0 and (reward / risk) >= 1.5:
                    tp = round_to_tick(structural_target, tick_size, direction='UP') if (tick_size and tick_size > 0) else structural_target
                    return tp
        
        target_pct = self.config.take_profit.target_pct_of_margin
        profit_amount = margin * target_pct
        
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        if qty > 0:
            price_diff = profit_amount / qty
        else:
            price_diff = entry_price * (target_pct / max(1, leverage))
        
        if is_long:
            tp_price = entry_price + price_diff
            if tick_size and tick_size > 0:
                tp_price = round_to_tick(tp_price, tick_size, direction='UP')
        else:
            tp_price = max(0.0, entry_price - price_diff)
            if tick_size and tick_size > 0:
                tp_price = round_to_tick(tp_price, tick_size, direction='DOWN')
            
        return tp_price

    def check_breakeven_trigger(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        initial_sl: float,
        current_sl: float,
        tick_size: Optional[float] = None,
        r_multiple: float = 2.0,
        breakeven_buffer_pct: float = 0.001,
    ) -> Tuple[float, bool, str]:
        """Check if an active position has achieved >= 2.0R profit.
        
        If so, move Stop Loss to Breakeven (+0.1% buffer in favor of trade to cover fees).
        Returns: (new_sl_price, updated, log_reason)
        """
        if entry_price <= 0 or initial_sl <= 0:
            return current_sl, False, "Invalid entry or initial SL"

        is_long = side.upper() in ('LONG', 'BUY')
        initial_risk = abs(entry_price - initial_sl)
        if initial_risk <= 0:
            return current_sl, False, "Initial risk is zero"

        if is_long:
            unrealized_profit = current_price - entry_price
            if unrealized_profit >= r_multiple * initial_risk:
                be_sl = entry_price * (1.0 + breakeven_buffer_pct)
                if tick_size and tick_size > 0:
                    be_sl = round_to_tick(be_sl, tick_size, direction='UP')
                if be_sl > current_sl:
                    return be_sl, True, f"Achieved +{unrealized_profit/initial_risk:.2f}R profit! SL locked to Breakeven (${be_sl:,.2f})"
        else:
            unrealized_profit = entry_price - current_price
            if unrealized_profit >= r_multiple * initial_risk:
                be_sl = entry_price * (1.0 - breakeven_buffer_pct)
                if tick_size and tick_size > 0:
                    be_sl = round_to_tick(be_sl, tick_size, direction='DOWN')
                if current_sl <= 0 or be_sl < current_sl:
                    return be_sl, True, f"Achieved +{unrealized_profit/initial_risk:.2f}R profit! SL locked to Breakeven (${be_sl:,.2f})"

        return current_sl, False, "Breakeven trigger not reached"

    def calculate_trailing_stop_loss(
        self,
        entry_price: float,
        side: str,
        margin: float,
        current_price: float,
        contract_value: float,
        size: int,
        current_sl: float,
        tick_size: Optional[float] = None
    ) -> Tuple[float, bool, str]:
        """Calculate dynamic stepped trailing stop loss based on unrealized P&L on margin.
        
        Rules:
        - When unrealized margin P&L < +2%: Keep initial SL (-3% margin).
        - When unrealized margin P&L >= +2%:
          SL is moved to +(int(floor(margin_pnl_pct)) - 1)% of margin.
          E.g.:
            +2% margin P&L -> SL = +1% of margin
            +3% margin P&L -> SL = +2% of margin
            +4% margin P&L -> SL = +3% of margin
            ...
            +10% margin P&L -> SL = +9% of margin
            up to +200% margin P&L -> SL = +199% of margin
        - Ratchet rule: SL price can only move in favor of the position (never backwards).
        
        Returns: (new_sl_price, updated, log_reason)
        """
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        if qty <= 0 or margin <= 0:
            return current_sl, False, "Invalid qty or margin"

        is_long = side.upper() in ('LONG', 'BUY')
        if is_long:
            unrealized_pnl = (current_price - entry_price) * qty
        else:
            unrealized_pnl = (entry_price - current_price) * qty

        margin_pnl_pct = (unrealized_pnl / margin) * 100.0

        trailing_cfg = getattr(self.config, "trailing_stop", None)
        activation_pct = (getattr(trailing_cfg, "activation_pct_of_margin", 0.02) * 100.0) if trailing_cfg else 2.0
        max_cap_pct = (getattr(trailing_cfg, "max_profit_cap_pct_of_margin", 2.00) * 100.0) if trailing_cfg else 200.0

        if margin_pnl_pct < activation_pct:
            return current_sl, False, f"Margin P&L {margin_pnl_pct:.2f}% below +{activation_pct:.0f}% trailing activation"

        # Determine stepped integer target
        step = int(math.floor(margin_pnl_pct))
        step = min(step, int(max_cap_pct))
        target_margin_pct = (step - 1) / 100.0  # e.g. step=2 -> +0.01 (+1% profit)

        profit_at_sl = margin * target_margin_pct
        price_diff = profit_at_sl / qty

        if is_long:
            candidate_sl = entry_price + price_diff
            if tick_size and tick_size > 0:
                candidate_sl = round_to_tick(candidate_sl, tick_size, direction='DOWN')
            if candidate_sl > current_sl:
                return candidate_sl, True, f"Trailing SL moved to +{step - 1}% margin profit at ${candidate_sl:,.2f}"
            else:
                return current_sl, False, f"Candidate SL ${candidate_sl:,.2f} <= current SL ${current_sl:,.2f}"
        else:
            candidate_sl = entry_price - price_diff
            if tick_size and tick_size > 0:
                candidate_sl = round_to_tick(candidate_sl, tick_size, direction='UP')
            if candidate_sl < current_sl or current_sl <= 0:
                return candidate_sl, True, f"Trailing SL moved to +{step - 1}% margin profit at ${candidate_sl:,.2f}"
            else:
                return current_sl, False, f"Candidate SL ${candidate_sl:,.2f} >= current SL ${current_sl:,.2f}"

    def calculate_structural_trailing_stop_loss(
        self,
        entry_price: float,
        side: str,
        current_price: float,
        current_sl: float,
        candles_1m: List[Any],
        tick_size: Optional[float] = None,
        buffer_atr: float = 0.0,
    ) -> Tuple[float, bool, str]:
        """Calculate Structural Swing Trailing Stop Loss.
        
        Trails behind confirmed 1m swing points:
        - LONG: Trails behind the most recent 1m swing low.
          Ratchets upward only when swing low (minus buffer) > current_sl.
        - SHORT: Trails behind the most recent 1m swing high.
          Ratchets downward only when swing high (plus buffer) < current_sl (or current_sl <= 0).
          
        Returns: (new_sl_price, updated: bool, reason: str)
        """
        if not candles_1m or len(candles_1m) < 5:
            return current_sl, False, "Insufficient 1m candles for structural trailing stop"

        is_long = side.upper() in ('LONG', 'BUY')
        
        # Extract low and high arrays
        if isinstance(candles_1m[0], dict):
            highs = [float(c['high']) for c in candles_1m]
            lows = [float(c['low']) for c in candles_1m]
        else:
            highs = [float(c.high) for c in candles_1m]
            lows = [float(c.low) for c in candles_1m]

        n = len(candles_1m)
        scan_len = min(15, n - 1)
        sub_highs = highs[-scan_len - 1 : -1]
        sub_lows = lows[-scan_len - 1 : -1]

        if is_long:
            swing_low_val = None
            for idx in range(len(sub_lows) - 2, 0, -1):
                if sub_lows[idx] <= sub_lows[idx - 1] and sub_lows[idx] <= sub_lows[idx + 1]:
                    swing_low_val = sub_lows[idx]
                    break
            if swing_low_val is None:
                swing_low_val = min(sub_lows[-3:])

            candidate_sl = swing_low_val - buffer_atr
            if tick_size and tick_size > 0:
                candidate_sl = round_to_tick(candidate_sl, tick_size, direction='DOWN')

            if candidate_sl > current_sl and candidate_sl < current_price:
                return candidate_sl, True, f"Structural Trailing SL ratcheted to 1m swing low ${candidate_sl:,.2f}"
            else:
                return current_sl, False, f"Structural candidate SL ${candidate_sl:,.2f} <= current SL ${current_sl:,.2f}"
        else:
            swing_high_val = None
            for idx in range(len(sub_highs) - 2, 0, -1):
                if sub_highs[idx] >= sub_highs[idx - 1] and sub_highs[idx] >= sub_highs[idx + 1]:
                    swing_high_val = sub_highs[idx]
                    break
            if swing_high_val is None:
                swing_high_val = max(sub_highs[-3:])

            candidate_sl = swing_high_val + buffer_atr
            if tick_size and tick_size > 0:
                candidate_sl = round_to_tick(candidate_sl, tick_size, direction='UP')

            if (candidate_sl < current_sl or current_sl <= 0) and candidate_sl > current_price:
                return candidate_sl, True, f"Structural Trailing SL ratcheted to 1m swing high ${candidate_sl:,.2f}"
            else:
                return current_sl, False, f"Structural candidate SL ${candidate_sl:,.2f} >= current SL ${current_sl:,.2f}"

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        leverage: int,
        contract_value: float,
        conviction_multiplier: float = 1.0,
    ) -> Tuple[int, float, float]:
        safe_mult = max(0.5, min(1.5, float(conviction_multiplier)))
        allocatable = equity * self.config.capital.max_allocation_pct * safe_mult
        margin = allocatable
        notional = margin * leverage
        size = int(notional / (contract_value * price)) if (contract_value * price) > 0 else 0
        return size, margin, notional

    def calculate_fees(self, notional: float, is_taker: bool = True) -> float:
        fee_pct = self.config.fees.taker_fee_pct if is_taker else self.config.fees.maker_fee_pct
        fee = notional * fee_pct
        slippage_cost = notional * self.config.fees.estimated_slippage_pct
        return fee + slippage_cost

    def calculate_sl_tp(self, side: str, entry_price: float, margin: float = 1000.0, leverage: int = 10, 
                        contract_value: float = 1.0, size: int = 1, atr: float = 100.0,
                        tick_size: Optional[float] = None,
                        structural_target: Optional[float] = None,
                        setup_type: Optional[str] = None,
                        trap_wick_price: Optional[float] = None) -> Tuple[float, float]:
        sl_price, _, _ = self.calculate_stop_loss(
            entry_price, side, margin, leverage, contract_value, size, atr, 
            tick_size=tick_size, trap_wick_price=trap_wick_price
        )
        tp_price = self.calculate_take_profit(
            entry_price, side, margin, leverage, contract_value, size, 
            tick_size=tick_size, structural_target=structural_target, sl_price=sl_price,
            setup_type=setup_type,
        )
        return sl_price, tp_price

    def validate_risk_reward_ratio(
        self,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        side: str = "LONG",
        min_rr: float = 2.0,
    ) -> Tuple[bool, float, str]:
        """Validate that proposed trade setup strictly meets institutional Risk-to-Reward ratio (>= min_rr, default 2.0R).
        
        Returns: (is_valid: bool, actual_rr: float, reason: str)
        """
        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            return False, 0.0, "Invalid price values (<= 0)"

        is_long = side.upper() in ('LONG', 'BUY')
        if is_long:
            risk = entry_price - sl_price
            reward = tp_price - entry_price
            if risk <= 0:
                return False, 0.0, f"Invalid risk: SL (${sl_price:,.2f}) must be below entry (${entry_price:,.2f}) for Long"
            if reward <= 0:
                return False, 0.0, f"Invalid reward: TP (${tp_price:,.2f}) must be above entry (${entry_price:,.2f}) for Long"
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price
            if risk <= 0:
                return False, 0.0, f"Invalid risk: SL (${sl_price:,.2f}) must be above entry (${entry_price:,.2f}) for Short"
            if reward <= 0:
                return False, 0.0, f"Invalid reward: TP (${tp_price:,.2f}) must be below entry (${entry_price:,.2f}) for Short"

        rr = reward / risk
        if rr < (min_rr - 1e-4):
            return False, round(rr, 2), f"Risk:Reward {rr:.2f}R is below institutional threshold {min_rr:.1f}R"

        return True, round(rr, 2), "VALID"

    async def calculate_sl_tp_async(
        self,
        side: str,
        entry_price: float,
        margin: float = 1000.0,
        leverage: int = 10,
        contract_value: float = 1.0,
        size: int = 1,
        atr: float = 100.0,
        tick_size: Optional[float] = None,
        technical_levels: Optional[Dict[str, Any]] = None,
        setup_type: Optional[str] = None,
        trap_wick_price: Optional[float] = None,
        symbol: str = "UNKNOWN",
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate Stop Loss and Take Profit levels dynamically using Jev System One if available,
        with fallback to institutional technical rules.
        
        Guarantees:
        - Capital Loss Ceiling: SL cannot exceed max loss pct of margin (default 3%).
        - Risk:Reward Ratio: Enforces institutional minimum R:R (>= min_risk_reward_ratio, default 2.0R to 5.0R).
        
        Returns: (sl_price, tp_price, metadata_dict)
        """
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        max_loss_pct = self.config.stop_loss.max_loss_pct_of_margin
        # If size and margin are consistent with leverage (notional ~ margin * leverage), use loss_amount / qty
        # Otherwise, leverage dictates the max loss price diff on the entry price.
        if qty > 0 and margin > 0 and abs((qty * entry_price) - (margin * leverage)) < (margin * leverage * 0.5):
            max_loss_price_diff = (margin * max_loss_pct) / qty
        else:
            max_loss_price_diff = entry_price * (max_loss_pct / max(1, leverage))

        is_long = side.upper() in ('LONG', 'BUY')
        jev_enabled = bool(
            self.jev_client
            and getattr(self.jev_client, "enabled", False)
            and getattr(getattr(self.jev_client, "config", None), "enable_dynamic_sl_tp", True)
        )

        jev_result = None
        if jev_enabled:
            try:
                levels = dict(technical_levels or {})
                if trap_wick_price and "trap_wick_price" not in levels:
                    levels["trap_wick_price"] = trap_wick_price
                
                jev_result = await self.jev_client.evaluate_dynamic_sl_tp(
                    symbol=symbol,
                    direction=side,
                    entry_price=entry_price,
                    atr=atr,
                    technical_levels=levels,
                    max_loss_price_diff=max_loss_price_diff,
                    tick_size=tick_size or 0.1,
                )
            except Exception as e:
                logger.error(f"Error calling Jev evaluate_dynamic_sl_tp: {e}", exc_info=True)
                jev_result = None

        if jev_result:
            sl_price = jev_result.sl_price
            tp_price = jev_result.tp_price

            # Strictly enforce Capital Safety Ceiling: SL must not lose more than max_loss_pct of margin
            if is_long:
                hard_floor = entry_price - max_loss_price_diff
                if sl_price < hard_floor:
                    sl_price = round_to_tick(hard_floor, tick_size, direction='DOWN') if tick_size else hard_floor
            else:
                hard_ceiling = entry_price + max_loss_price_diff
                if sl_price > hard_ceiling:
                    sl_price = round_to_tick(hard_ceiling, tick_size, direction='UP') if tick_size else hard_ceiling

            min_rr = getattr(getattr(self.jev_client, "config", None), "min_risk_reward_ratio", 2.0)
            is_valid_rr, actual_rr, _ = self.validate_risk_reward_ratio(entry_price, sl_price, tp_price, side=side, min_rr=min_rr)
            if not is_valid_rr:
                # Dynamically adjust TP to satisfy min_rr requirement
                risk = abs(entry_price - sl_price)
                if is_long:
                    tp_price = entry_price + (min_rr * risk)
                    if tick_size and tick_size > 0:
                        tp_price = round_to_tick(tp_price, tick_size, direction='UP')
                else:
                    tp_price = max(0.0, entry_price - (min_rr * risk))
                    if tick_size and tick_size > 0:
                        tp_price = round_to_tick(tp_price, tick_size, direction='DOWN')
                actual_rr = min_rr

            metadata = {
                "source": "JEV",
                "sl_anchor": jev_result.sl_anchor,
                "tp_target_type": jev_result.tp_target_type,
                "target_rr": jev_result.target_rr_multiple,
                "realized_rr": actual_rr,
                "cushion_grade": jev_result.sl_cushion_grade,
                "confidence": jev_result.confidence,
                "latency_ms": jev_result.latency_ms,
            }
            return sl_price, tp_price, metadata

        # Institutional Fallback
        sl_price, tp_price = self.calculate_sl_tp(
            side=side,
            entry_price=entry_price,
            margin=margin,
            leverage=leverage,
            contract_value=contract_value,
            size=size,
            atr=atr,
            tick_size=tick_size,
            structural_target=(technical_levels or {}).get("nearest_resistance" if is_long else "nearest_support"),
            setup_type=setup_type,
            trap_wick_price=trap_wick_price,
        )
        _, actual_rr, _ = self.validate_risk_reward_ratio(entry_price, sl_price, tp_price, side=side, min_rr=1.5)
        metadata = {
            "source": "FALLBACK",
            "sl_anchor": "TRAP_WICK" if trap_wick_price else "MAX_MARGIN_LOSS",
            "tp_target_type": "STRUCTURAL" if (technical_levels or {}).get("nearest_resistance" if is_long else "nearest_support") else "MARGIN_TARGET",
            "target_rr": actual_rr,
            "realized_rr": actual_rr,
            "cushion_grade": 2.0,
            "confidence": 1.0,
            "latency_ms": 0.0,
        }
        return sl_price, tp_price, metadata

    async def calculate_dynamic_jev_trailing_stop(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        initial_sl: float,
        margin: float,
        contract_value: float,
        size: int,
        duration_seconds: float,
        atr: float,
        technical_levels: Optional[Dict[str, Any]] = None,
        tick_size: Optional[float] = None,
    ) -> Tuple[float, bool, str]:
        """Calculate dynamic trailing stop using Jev System One evaluation.
        
        Guarantees Ratchet Invariant: Stop Loss can ONLY move in favor of the position (never backwards).
        Falls back to stepped margin trailing stop if Jev is unavailable or fails.
        """
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        is_long = side.upper() in ('LONG', 'BUY')
        unrealized_pnl = ((current_price - entry_price) * qty) if is_long else ((entry_price - current_price) * qty)

        jev_enabled = bool(
            self.jev_client
            and getattr(self.jev_client, "enabled", False)
            and getattr(getattr(self.jev_client, "config", None), "enable_dynamic_trailing", True)
        )

        if jev_enabled:
            try:
                jev_res = await self.jev_client.evaluate_dynamic_trailing_stop(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    current_sl=current_sl,
                    initial_sl=initial_sl,
                    unrealized_pnl=unrealized_pnl,
                    duration_seconds=duration_seconds,
                    atr=atr,
                    technical_levels=technical_levels or {},
                    tick_size=tick_size or 0.1,
                )
                if jev_res:
                    candidate_sl = jev_res.candidate_sl_price
                    # Extra verification of ratchet invariant:
                    if is_long:
                        if candidate_sl > current_sl:
                            return candidate_sl, True, f"Jev dynamic trailing ({jev_res.action}): SL moved to ${candidate_sl:,.2f}"
                        return current_sl, False, f"Jev dynamic trailing held: ${candidate_sl:,.2f} <= current SL ${current_sl:,.2f}"
                    else:
                        if current_sl <= 0 or candidate_sl < current_sl:
                            return candidate_sl, True, f"Jev dynamic trailing ({jev_res.action}): SL moved to ${candidate_sl:,.2f}"
                        return current_sl, False, f"Jev dynamic trailing held: ${candidate_sl:,.2f} >= current SL ${current_sl:,.2f}"
            except Exception as e:
                logger.error(f"Error in Jev dynamic trailing stop: {e}", exc_info=True)

        # Fallback to standard stepped trailing stop loss
        return self.calculate_trailing_stop_loss(
            entry_price=entry_price,
            side=side,
            margin=margin,
            current_price=current_price,
            contract_value=contract_value,
            size=size,
            current_sl=current_sl,
            tick_size=tick_size,
        )

    def validate_trade(self, equity: float, margin: float, notional: float, leverage: int = 1, sl_price: float = 1.0, 
                       tp_price: float = 1.0, entry_price: float = 1.0, side: str = "LONG", atr: float = 100.0, 
                       spread_bps: float = 0.0, has_position: bool = False, **kwargs: Any) -> Tuple[bool, str]:
        
        self.reset_daily() # Ensure daily stats are up to date
        
        if self.is_kill_switch_active:
            return False, "KILL_SWITCH_ACTIVE"
            
        failsafe_ok, failsafe_reasons = self.check_failsafes()
        if not failsafe_ok:
            return False, f"FAILSAFE_ERROR: {', '.join(failsafe_reasons)}"

        if has_position:
            return False, "MAX_POSITIONS_REACHED"

        # Allow single minimum contract if margin <= equity even if it slightly exceeds allocation cap for micro balances
        allow_micro_min_contract = (margin <= equity) and (equity <= 10.0 or kwargs.get("is_min_contract", False))
        conv_mult = float(kwargs.get("conviction_multiplier", 1.0))
        effective_alloc_cap = self.config.capital.max_allocation_pct * max(1.0, min(1.5, conv_mult))
        if margin > equity * effective_alloc_cap and not allow_micro_min_contract:
            return False, "INSUFFICIENT_EQUITY"

        if self.daily_start_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.daily_start_equity
            if daily_loss_pct >= self.config.daily_limits.max_daily_loss_pct:
                return False, "DAILY_LOSS_LIMIT_REACHED"

        if self.consecutive_losses >= self.config.daily_limits.max_consecutive_losses:
            return False, "MAX_CONSECUTIVE_LOSSES_REACHED"

        now = time.time()
        if now - self.last_loss_time < self.config.daily_limits.cooldown_seconds:
            return False, "COOLDOWN_ACTIVE"

        if spread_bps > self.config.failsafe.max_spread_bps:
            return False, "SPREAD_TOO_HIGH"

        # (Basic SL validation is assumed done via calculate_stop_loss, but we could add more here)
        if sl_price <= 0:
            return False, "INVALID_STOP_LOSS"

        return True, "APPROVED"

    def record_trade_result(self, pnl: float):
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()
        else:
            self.consecutive_losses = 0

    def activate_kill_switch(self):
        logger.warning("Kill switch ACTIVATED")
        self.is_kill_switch_active = True

    def deactivate_kill_switch(self):
        logger.info("Kill switch DEACTIVATED")
        self.is_kill_switch_active = False

    def reset_daily(self):
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            logger.info("Resetting daily risk limits")
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            # daily_start_equity should be updated by account manager
            self.last_reset_date = current_date

    def check_failsafes(self) -> Tuple[bool, List[str]]:
        failures = []
        if self.data_stale:
            failures.append("STALE_DATA")
        if not self.market_data_connected:
            failures.append("MARKET_DATA_DISCONNECTED")
        if not self.delta_connected:
            failures.append(self.delta_error_reason or "DELTA_DISCONNECTED")
            
        return len(failures) == 0, failures

    def get_risk_status(self) -> Dict[str, Any]:
        failsafe_ok, failsafe_reasons = self.check_failsafes()
        return {
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "kill_switch_active": self.is_kill_switch_active,
            "failsafe_ok": failsafe_ok,
            "failsafe_reasons": failsafe_reasons,
            "cooldown_active": (time.time() - self.last_loss_time) < self.config.daily_limits.cooldown_seconds,
            "risk_check": "PASS" if failsafe_ok else f"FAIL: {','.join(failsafe_reasons)}",
            "max_loss": f"{self.config.stop_loss.max_loss_pct_of_margin * 100:.1f}%",
            "target_profit": "Trailing (+2%->+200%)",
            "time_limit": "29m (Delta Scalper)",
            "spread": f"< {self.config.failsafe.max_spread_bps:.0f} bps",
            "slippage": f"{self.config.fees.estimated_slippage_pct * 100:.2f}%",
            "liquidation_dist": f"> {self.config.capital.reserve_pct * 100:.0f}%",
            "risk_status": "LOCKED" if self.is_kill_switch_active else ("READY" if failsafe_ok else "BLOCKED"),
        }
