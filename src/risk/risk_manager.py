import math
import time
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timezone

from src.config import RiskConfig
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
    def __init__(self, config: RiskConfig):
        self.config = config
        
        # State tracking
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_loss_time: float = 0.0
        self.daily_start_equity: float = 0.0
        self.is_kill_switch_active: bool = False
        
        # Failsafe flags
        self.data_stale: bool = False
        self.binance_connected: bool = True
        self.delta_connected: bool = True
        self.delta_error_reason: str = ""
        
        # Keep track of last daily reset
        self.last_reset_date = datetime.now(timezone.utc).date()

    def validate_leverage(self, symbol_or_leverage: Any, exchange_max_leverage: float) -> Tuple[bool, int, str]:
        if isinstance(symbol_or_leverage, (int, float)):
            desired_leverage = int(symbol_or_leverage)
            if desired_leverage > exchange_max_leverage:
                return False, int(exchange_max_leverage), f"Leverage {desired_leverage} exceeds max {exchange_max_leverage}"
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
                            tick_size: Optional[float] = None) -> Tuple[float, bool, str]:
        
        max_loss_pct = self.config.stop_loss.max_loss_pct_of_margin
        loss_amount = margin * max_loss_pct
        
        # price_diff * size * contract_value = loss_amount
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        if qty > 0:
            price_diff = loss_amount / qty
        else:
            price_diff = entry_price * (max_loss_pct / max(1, leverage))
        
        if side.upper() in ('LONG', 'BUY'):
            sl_price = max(0.0, entry_price - price_diff)
            if tick_size and tick_size > 0:
                sl_price = round_to_tick(sl_price, tick_size, direction='DOWN')
        else:
            sl_price = entry_price + price_diff
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
                              tick_size: Optional[float] = None) -> float:
        
        target_pct = self.config.take_profit.target_pct_of_margin
        profit_amount = margin * target_pct
        
        qty = (size * contract_value) if (size > 0 and contract_value > 0) else 0.0
        if qty > 0:
            price_diff = profit_amount / qty
        else:
            price_diff = entry_price * (target_pct / max(1, leverage))
        
        if side.upper() in ('LONG', 'BUY'):
            tp_price = entry_price + price_diff
            if tick_size and tick_size > 0:
                tp_price = round_to_tick(tp_price, tick_size, direction='UP')
        else:
            tp_price = max(0.0, entry_price - price_diff)
            if tick_size and tick_size > 0:
                tp_price = round_to_tick(tp_price, tick_size, direction='DOWN')
            
        return tp_price

    def calculate_position_size(self, equity: float, price: float, leverage: int, contract_value: float) -> Tuple[int, float, float]:
        allocatable = equity * self.config.capital.max_allocation_pct
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
                        tick_size: Optional[float] = None) -> Tuple[float, float]:
        sl_price, _, _ = self.calculate_stop_loss(entry_price, side, margin, leverage, contract_value, size, atr, tick_size=tick_size)
        tp_price = self.calculate_take_profit(entry_price, side, margin, leverage, contract_value, size, tick_size=tick_size)
        return sl_price, tp_price

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

        if margin > equity * self.config.capital.max_allocation_pct:
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
        if not self.binance_connected:
            failures.append("BINANCE_DISCONNECTED")
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
            "target_profit": f"{self.config.take_profit.target_pct_of_margin * 100:.1f}%",
            "spread": f"< {self.config.failsafe.max_spread_bps:.0f} bps",
            "slippage": f"{self.config.fees.estimated_slippage_pct * 100:.2f}%",
            "liquidation_dist": f"> {self.config.capital.reserve_pct * 100:.0f}%",
            "risk_status": "LOCKED" if self.is_kill_switch_active else ("READY" if failsafe_ok else "BLOCKED"),
        }
