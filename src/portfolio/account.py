import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import orjson

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    entry_time: float
    size: int
    leverage: int
    margin: float
    notional: float
    sl: float
    tp: float
    unrealized_pnl: float = 0.0
    current_price: float = 0.0
    contract_value: float = 1.0

CONTRACT_VALUES: Dict[str, float] = {
    "BTCUSD": 0.001,
    "ETHUSD": 0.01,
}

class AccountManager:
    def __init__(self, is_paper: bool = False, initial_paper_balance: float = 10000.0, paper_balance: Optional[float] = None):
        self.is_paper = is_paper
        bal = paper_balance if paper_balance is not None else initial_paper_balance
        self.equity: float = bal if is_paper else 0.0
        self.available_margin: float = bal if is_paper else 0.0
        self.used_margin: float = 0.0
        self.reserve: float = (self.equity * 0.20) if is_paper else 0.0
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = self.equity
        self.contract_values: Dict[str, float] = dict(CONTRACT_VALUES)
        
        self.positions: Dict[str, Position] = {}

    def get_contract_value(self, symbol: str) -> float:
        if symbol in self.contract_values:
            return self.contract_values[symbol]
        if "BTC" in symbol:
            return 0.001
        if "ETH" in symbol:
            return 0.01
        return 1.0
        
    def update_from_exchange(self, balances: Dict[str, Any], exchange_positions: List[Dict[str, Any]]):
        if self.is_paper:
            return
            
        # Update logic based on Delta API response format
        self.equity = float(balances.get('equity', self.equity))
        self.available_margin = float(balances.get('available_margin', balances.get('available_balance', self.available_margin)))
        self.used_margin = float(balances.get('position_margin', balances.get('used_margin', balances.get('margin', self.used_margin))))
        self.reserve = self.equity * 0.20
        
        # Update positions while preserving contract_value, leverage, SL, and TP
        new_positions = {}
        for p in exchange_positions:
            symbol = p.get('product_symbol') or p.get('symbol')
            size_raw = float(p.get('size', 0.0))
            if symbol and abs(size_raw) > 0:
                cv = self.get_contract_value(symbol)
                entry_p = float(p.get('entry_price', 0.0))
                size_abs = abs(int(size_raw))
                notional = size_abs * cv * entry_p
                
                existing = self.positions.get(symbol)
                lev = existing.leverage if (existing and existing.leverage > 1) else int(float(p.get('leverage', 150) or 150))
                sl_val = existing.sl if (existing and existing.sl > 0) else float(p.get('sl', 0.0))
                tp_val = existing.tp if (existing and existing.tp > 0) else float(p.get('tp', 0.0))
                entry_t = existing.entry_time if existing else float(p.get('entry_time', time.time()))
                margin_val = float(p.get('margin', p.get('position_margin', 0.0)))
                if margin_val <= 0 and lev > 0:
                    margin_val = notional / lev
                
                new_positions[symbol] = Position(
                    symbol=symbol,
                    side='LONG' if size_raw > 0 else 'SHORT',
                    entry_price=entry_p,
                    entry_time=entry_t,
                    size=size_abs,
                    leverage=lev,
                    margin=margin_val,
                    notional=notional,
                    sl=sl_val,
                    tp=tp_val,
                    unrealized_pnl=float(p.get('unrealized_pnl', 0.0)) if ('unrealized_pnl' in p and p['unrealized_pnl'] is not None) else (existing.unrealized_pnl if existing else 0.0),
                    current_price=existing.current_price if existing else entry_p,
                    contract_value=cv,
                )

        self.positions = new_positions


    def update_unrealized_pnl(self, symbol: str, current_price: float, contract_value: Optional[float] = None):
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            cv = contract_value if (contract_value is not None and contract_value != 1.0) else (pos.contract_value or self.get_contract_value(symbol))
            pos.contract_value = cv
            if pos.side == 'LONG':
                pnl = (current_price - pos.entry_price) * pos.size * cv
            else:
                pnl = (pos.entry_price - current_price) * pos.size * cv
            pos.unrealized_pnl = pnl

    def update_current_price(self, symbol: str, current_price: float):
        """Update the current live price for a position and recalculate unrealized P&L."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            cv = pos.contract_value or self.get_contract_value(symbol)
            pos.contract_value = cv
            if pos.side == 'LONG':
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size * cv
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size * cv

    def update_paper_position(
        self,
        symbol: str,
        side: str,
        size: int,
        price: float,
        leverage: int,
        margin: float,
        notional: float,
        sl: float,
        tp: float,
    ) -> None:
        if not self.is_paper:
            return
        if size == 0:
            self.positions.pop(symbol, None)
            self.used_margin = sum(p.margin for p in self.positions.values())
            self.available_margin = max(0.0, self.equity - self.used_margin)
            return

        cv = self.get_contract_value(symbol)
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            entry_time=time.time(),
            size=size,
            leverage=leverage,
            margin=margin,
            notional=notional,
            sl=sl,
            tp=tp,
            unrealized_pnl=0.0,
            current_price=price,
            contract_value=cv,
        )
        self.used_margin = sum(p.margin for p in self.positions.values())
        self.available_margin = max(0.0, self.equity - self.used_margin)

    def close_paper_position(self, symbol: str, close_price: float, contract_value: Optional[float] = None) -> float:
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            cv = contract_value if (contract_value is not None and contract_value != 1.0) else (pos.contract_value or self.get_contract_value(symbol))
            if pos.side == 'LONG':
                pnl = (close_price - pos.entry_price) * pos.size * cv
            else:
                pnl = (pos.entry_price - close_price) * pos.size * cv
            self.equity += pnl
            self.daily_pnl += pnl
            self.used_margin = sum(p.margin for p in self.positions.values())
            self.available_margin = max(0.0, self.equity - self.used_margin)
            self.reserve = self.equity * 0.20
            return pnl
        return 0.0

    def get_holding_time(self, symbol: str) -> Optional[timedelta]:
        if symbol in self.positions:
            entry_time = self.positions[symbol].entry_time
            return timedelta(seconds=time.time() - entry_time)
        return None

    def reset_daily(self, current_equity: Optional[float] = None):
        if current_equity is not None:
            self.daily_start_equity = current_equity
        else:
            self.daily_start_equity = self.equity
        self.daily_pnl = 0.0

    def get_position_dict(self) -> Dict[str, Any]:
        if not self.positions:
            return {}
        pos = next(iter(self.positions.values()))
        holding_sec = time.time() - pos.entry_time
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry": pos.entry_price,
            "current_price": pos.current_price if pos.current_price > 0 else pos.entry_price,
            "leverage": pos.leverage,
            "margin": pos.margin,
            "notional": pos.notional,
            "sl": pos.sl,
            "tp": pos.tp,
            "unrealized_pnl": pos.unrealized_pnl,
            "margin_pnl_pct": (pos.unrealized_pnl / pos.margin * 100) if pos.margin > 0 else 0.0,
            "holding_time": f"{int(holding_sec)}s",
        }

    def to_dashboard_dict(self) -> Dict[str, Any]:
        used_margin_pct = (self.used_margin / self.equity * 100.0) if self.equity > 0 else 0.0
        reserve_pct = (self.reserve / self.equity * 100.0) if self.equity > 0 else 20.0
        daily_loss_limit = -0.03 * self.daily_start_equity if self.daily_start_equity > 0 else -300.0
        return {
            "equity": self.equity,
            "available_margin": self.available_margin,
            "used_margin": self.used_margin,
            "used_margin_pct": used_margin_pct,
            "reserve_pct": reserve_pct,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": daily_loss_limit,
            "positions": [asdict(p) for p in self.positions.values()],
            "is_paper": self.is_paper
        }

    def save_state(self, path: str):
        state = {
            "equity": self.equity,
            "available_margin": self.available_margin,
            "used_margin": self.used_margin,
            "daily_pnl": self.daily_pnl,
            "daily_start_equity": self.daily_start_equity,
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()}
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2))
        logger.info(f"Account state saved to {path}")

    def load_state(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"State file {path} not found")
            return
            
        with open(path, "rb") as f:
            state = orjson.loads(f.read())
            
        self.equity = state.get("equity", self.equity)
        self.available_margin = state.get("available_margin", self.available_margin)
        self.used_margin = state.get("used_margin", self.used_margin)
        self.daily_pnl = state.get("daily_pnl", self.daily_pnl)
        self.daily_start_equity = state.get("daily_start_equity", self.daily_start_equity)
        
        pos_data = state.get("positions", {})
        self.positions = {}
        for sym, data in pos_data.items():
            self.positions[sym] = Position(**data)
            
        logger.info(f"Account state loaded from {path}")
