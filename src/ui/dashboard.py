"""
Dashboard UI for the trading bot using Rich.
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Awaitable, Optional
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

from src.utils.logger import get_logger

logger = get_logger(__name__)

class Dashboard:
    def __init__(self, mode: str, target_leverage: int = 100):
        self.mode = mode.upper()
        self.target_leverage = target_leverage
        self.console = Console()
        self.account_data: Dict[str, Any] = {}
        self.position_data: Dict[str, Any] = {}
        self.signal_data: Dict[str, Any] = {}
        self.risk_data: Dict[str, Any] = {}
        self.execution_data: Dict[str, Any] = {}
        self.market_watch_data: Dict[str, Any] = {}

    def update(
        self,
        account_data: Dict[str, Any],
        position_data: Dict[str, Any],
        signal_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        execution_data: Dict[str, Any],
        market_watch_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.account_data = account_data
        self.position_data = position_data
        self.signal_data = signal_data
        self.risk_data = risk_data
        self.execution_data = execution_data
        if market_watch_data is not None:
            self.market_watch_data = market_watch_data

    def _build_account_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        equity = self.account_data.get("equity", 0.0)
        table.add_row("Equity", f": ${equity:,.2f}")
        
        available_margin = self.account_data.get("available_margin", 0.0)
        table.add_row("Available Margin", f": ${available_margin:,.2f}")
        
        used_margin_pct = self.account_data.get("used_margin_pct", 0.0)
        table.add_row("Used Margin %", f": {used_margin_pct:.1f}%")
        
        reserve_pct = self.account_data.get("reserve_pct", 0.0)
        table.add_row("Reserve %", f": {reserve_pct:.1f}%")
        
        daily_pnl = self.account_data.get("daily_pnl", 0.0)
        color = "green" if daily_pnl >= 0 else "red"
        table.add_row("Daily P&L", Text(f": ${daily_pnl:,.2f}", style=color))
        
        daily_loss_limit = self.account_data.get("daily_loss_limit", 0.0)
        table.add_row("Daily Loss Limit", f": -${abs(daily_loss_limit):,.2f}")
        
        return Panel(table, title="ACCOUNT", border_style="blue")

    def _build_market_watch_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")

        assets = self.market_watch_data.get("assets", {})
        btc = assets.get("BTCUSD", {})
        eth = assets.get("ETHUSD", {})

        # BTC Live Stats
        btc_delta = btc.get("delta_price") or btc.get("price") or self.signal_data.get("btc_price", 0.0)
        btc_delta_str = f"${btc_delta:,.2f}" if (isinstance(btc_delta, (int, float)) and btc_delta > 0) else "--"
        btc_bias = btc.get("bias_15m") or "--"
        btc_bias_col = "green" if btc_bias == "BULLISH" else ("red" if btc_bias == "BEARISH" else "yellow")
        table.add_row("BTC/USD Live", Text(f": {btc_delta_str}", style="bold green"))

        btc_pdh = btc.get("pdh", 0.0)
        btc_pdl = btc.get("pdl", 0.0)
        btc_poc = btc.get("poc", 0.0)
        if btc_pdh or btc_pdl:
            htf_str = f"PDH=${btc_pdh:,.2f} | PDL=${btc_pdl:,.2f}" + (f" | POC=${btc_poc:,.2f}" if btc_poc else "")
            table.add_row("BTC HTF Levels", Text(f": {htf_str}", style="cyan"))

        btc_vah = btc.get("vah", 0.0)
        btc_val = btc.get("val", 0.0)
        if btc_vah or btc_val:
            table.add_row("BTC Value Area", Text(f": VAH=${btc_vah:,.2f} | VAL=${btc_val:,.2f}", style="cyan"))

        btc_eqh = btc.get("eqh", 0.0)
        btc_eql = btc.get("eql", 0.0)
        btc_ah = btc.get("asian_high", 0.0)
        btc_al = btc.get("asian_low", 0.0)
        btc_liq = []
        if btc_eqh: btc_liq.append(f"EQH=${btc_eqh:,.2f}")
        if btc_eql: btc_liq.append(f"EQL=${btc_eql:,.2f}")
        if btc_ah: btc_liq.append(f"AsianH=${btc_ah:,.2f}")
        if btc_al: btc_liq.append(f"AsianL=${btc_al:,.2f}")
        if btc_liq:
            table.add_row("BTC Liquidity", Text(f": {' | '.join(btc_liq)}", style="magenta"))

        if btc.get("ob_detected"):
            ob_d = btc.get("ob_direction", "NONE")
            ob_b = btc.get("ob_bottom", 0.0)
            ob_t = btc.get("ob_top", 0.0)
            mit_str = " (Testing)" if btc.get("ob_testing") else ""
            table.add_row("BTC Order Block", Text(f": {ob_d} [${ob_b:,.2f}-${ob_t:,.2f}]{mit_str}", style="bold green" if ob_d == "BULLISH" else "bold red"))

        if btc.get("squeeze_fired"):
            table.add_row("BTC Squeeze", Text(": SQUEEZE FIRED! (Expansion)", style="bold green"))
        elif btc.get("is_squeeze"):
            table.add_row("BTC Squeeze", Text(": SQUEEZE ACTIVE (Compression)", style="bold yellow"))

        btc_s = btc.get("support", 0.0)
        btc_r = btc.get("resistance", 0.0)
        if btc_s or btc_r:
            sr_str = f"S=${btc_s:,.2f} | R=${btc_r:,.2f}" if btc_s and btc_r else (f"S=${btc_s:,.2f}" if btc_s else f"R=${btc_r:,.2f}")
            table.add_row("BTC Key S/R", Text(f": {sr_str}", style="yellow"))

        btc_bos = btc.get("bos_5m", False)
        btc_setup = btc.get("setup_type", "NONE")
        btc_sub = f"15M={btc_bias} | BOS={btc_bos}"
        if btc_setup and btc_setup != "NONE":
            btc_sub += f" | {btc_setup}"
        table.add_row("BTC Trend/5M", Text(f": {btc_sub}", style=btc_bias_col))

        # ETH Live Stats
        eth_delta = eth.get("delta_price") or eth.get("price") or self.signal_data.get("eth_price", 0.0)
        eth_delta_str = f"${eth_delta:,.2f}" if (isinstance(eth_delta, (int, float)) and eth_delta > 0) else "--"
        eth_bias = eth.get("bias_15m") or "--"
        eth_bias_col = "green" if eth_bias == "BULLISH" else ("red" if eth_bias == "BEARISH" else "yellow")
        table.add_row("ETH/USD Live", Text(f": {eth_delta_str}", style="bold green"))

        eth_pdh = eth.get("pdh", 0.0)
        eth_pdl = eth.get("pdl", 0.0)
        eth_poc = eth.get("poc", 0.0)
        if eth_pdh or eth_pdl:
            htf_str = f"PDH=${eth_pdh:,.2f} | PDL=${eth_pdl:,.2f}" + (f" | POC=${eth_poc:,.2f}" if eth_poc else "")
            table.add_row("ETH HTF Levels", Text(f": {htf_str}", style="cyan"))

        eth_vah = eth.get("vah", 0.0)
        eth_val = eth.get("val", 0.0)
        if eth_vah or eth_val:
            table.add_row("ETH Value Area", Text(f": VAH=${eth_vah:,.2f} | VAL=${eth_val:,.2f}", style="cyan"))

        eth_eqh = eth.get("eqh", 0.0)
        eth_eql = eth.get("eql", 0.0)
        eth_ah = eth.get("asian_high", 0.0)
        eth_al = eth.get("asian_low", 0.0)
        eth_liq = []
        if eth_eqh: eth_liq.append(f"EQH=${eth_eqh:,.2f}")
        if eth_eql: eth_liq.append(f"EQL=${eth_eql:,.2f}")
        if eth_ah: eth_liq.append(f"AsianH=${eth_ah:,.2f}")
        if eth_al: eth_liq.append(f"AsianL=${eth_al:,.2f}")
        if eth_liq:
            table.add_row("ETH Liquidity", Text(f": {' | '.join(eth_liq)}", style="magenta"))

        if eth.get("ob_detected"):
            ob_d = eth.get("ob_direction", "NONE")
            ob_b = eth.get("ob_bottom", 0.0)
            ob_t = eth.get("ob_top", 0.0)
            mit_str = " (Testing)" if eth.get("ob_testing") else ""
            table.add_row("ETH Order Block", Text(f": {ob_d} [${ob_b:,.2f}-${ob_t:,.2f}]{mit_str}", style="bold green" if ob_d == "BULLISH" else "bold red"))

        if eth.get("squeeze_fired"):
            table.add_row("ETH Squeeze", Text(": SQUEEZE FIRED! (Expansion)", style="bold green"))
        elif eth.get("is_squeeze"):
            table.add_row("ETH Squeeze", Text(": SQUEEZE ACTIVE (Compression)", style="bold yellow"))

        eth_s = eth.get("support", 0.0)
        eth_r = eth.get("resistance", 0.0)
        if eth_s or eth_r:
            sr_str = f"S=${eth_s:,.2f} | R=${eth_r:,.2f}" if eth_s and eth_r else (f"S=${eth_s:,.2f}" if eth_s else f"R=${eth_r:,.2f}")
            table.add_row("ETH Key S/R", Text(f": {sr_str}", style="yellow"))

        eth_bos = eth.get("bos_5m", False)
        eth_setup = eth.get("setup_type", "NONE")
        eth_sub = f"15M={eth_bias} | BOS={eth_bos}"
        if eth_setup and eth_setup != "NONE":
            eth_sub += f" | {eth_setup}"
        table.add_row("ETH Trend/5M", Text(f": {eth_sub}", style=eth_bias_col))

        # Strategic status
        table.add_row("Target Lev", f": {self.target_leverage}x Isolated")
        table.add_row("Margin Target", ": -3.0% SL | Trailing (+2%->+1% ... +200%)")
        table.add_row("Delta Offer", ": 29m Limit (Zero Closing Fee)")

        trigger = self.market_watch_data.get("next_trigger") or self.signal_data.get("next_trigger", "Awaiting 5M BOS/CHoCH + ML >= 65%")
        trigger_col = "green" if "READY" in trigger.upper() else "yellow"
        table.add_row("Trigger State", Text(f": {trigger}", style=trigger_col))

        return Panel(table, title="REAL-TIME MARKET WATCH", border_style="cyan")

    def _build_position_panel(self) -> Panel:
        has_active_pos = bool(
            self.position_data
            and self.position_data.get('symbol')
            and self.position_data.get('symbol') != '--'
        )
        if not has_active_pos:
            return self._build_market_watch_panel()

        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        sym = self.position_data.get('symbol', '--')
        table.add_row("Symbol", f": {sym}")
        table.add_row("Side", f": {self.position_data.get('side', '--')}")
        
        entry = self.position_data.get('entry', '--')
        entry_val = entry if isinstance(entry, (int, float)) else 0.0
        table.add_row("Entry", f": ${entry_val:,.2f}" if entry_val > 0 else f": {entry}")
        
        cur = self.position_data.get('current_price', '--')
        cur_val = cur if isinstance(cur, (int, float)) else entry_val
        table.add_row("Current", f": ${cur_val:,.2f}" if cur_val > 0 else f": {cur}")
        
        lev = self.position_data.get('leverage')
        lev_val = int(lev) if (isinstance(lev, (int, float)) and lev > 1) else self.target_leverage
        table.add_row("Leverage", f": {lev_val}x")
        
        margin = self.position_data.get('margin', '--')
        margin_val = margin if isinstance(margin, (int, float)) else 0.0
        table.add_row("Margin", f": ${margin_val:,.2f}" if margin_val > 0 else f": {margin}")
        
        notional = self.position_data.get('notional', 0.0)
        notional_val = notional if isinstance(notional, (int, float)) else 0.0
        if notional_val <= 0.0 and margin_val > 0:
            notional_val = margin_val * lev_val
        table.add_row("Notional", f": ${notional_val:,.2f}" if notional_val > 0 else ": --")
        
        sl = self.position_data.get('sl', 0.0)
        sl_val = sl if isinstance(sl, (int, float)) else 0.0
        if sl_val <= 0.0 and entry_val > 0:
            is_long = str(self.position_data.get('side', '')).upper() in ('BUY', 'LONG')
            sl_val = entry_val * (1.0 - 0.03 / lev_val) if is_long else entry_val * (1.0 + 0.03 / lev_val)
        
        sl_str = f"${sl_val:,.2f}" if sl_val > 0 else "--"
        if sl_val > 0 and entry_val > 0:
            is_long = str(self.position_data.get('side', '')).upper() in ('BUY', 'LONG')
            if (is_long and sl_val > entry_val) or (not is_long and sl_val < entry_val):
                table.add_row("SL (Trailed)", Text(f": {sl_str} (Profit Locked)", style="bold green"))
            else:
                table.add_row("SL", f": {sl_str}")
        else:
            table.add_row("SL", f": {sl_str}")
        
        tp = self.position_data.get('tp', '--')
        table.add_row("TP", f": {tp} (Dynamic Trailing)" if tp in ('--', 0, 0.0, None) else (f": ${tp:,.2f}" if isinstance(tp, (int, float)) else f": {tp}"))
        
        pnl = self.position_data.get("unrealized_pnl", "--")
        if isinstance(pnl, (int, float)):
            color = "green" if pnl >= 0 else "red"
            table.add_row("Unrealized P&L", Text(f": ${pnl:,.2f}", style=color))
        else:
            table.add_row("Unrealized P&L", f": {pnl}")
            
        margin_pnl_pct = self.position_data.get('margin_pnl_pct', '--')
        if isinstance(margin_pnl_pct, (int, float)):
            pct_color = "green" if margin_pnl_pct >= 0 else "red"
            table.add_row("Margin P&L %", Text(f": {margin_pnl_pct:+.2f}%", style=pct_color))
        else:
            table.add_row("Margin P&L %", f": {margin_pnl_pct}")

        ht_str = str(self.position_data.get('holding_time', '--'))
        table.add_row("Holding Time", f": {ht_str}")

        setup_p = self.position_data.get('setup_type', '')
        if setup_p in ("HTF_BREAKOUT", "BREAKOUT_RETEST"):
            table.add_row("Mode", Text(f": RUNNER ({setup_p} 2.5R+ Target)", style="bold magenta"))
        else:
            # Delta Scalper 29m timer
            holding_sec = 0.0
            if ht_str.endswith('s') and ht_str[:-1].isdigit():
                holding_sec = float(ht_str[:-1])
            remaining = max(0.0, 1740.0 - holding_sec)
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            rem_col = "green" if remaining > 300 else ("yellow" if remaining > 60 else "bold red")
            table.add_row("Delta Scalper", Text(f": {rem_min}m {rem_sec:02d}s left (Zero Fee)", style=rem_col))

        # Support/Resistance levels
        sup = self.position_data.get('nearest_support', 0)
        res = self.position_data.get('nearest_resistance', 0)
        if sup or res:
            sr_text = f": S=${sup:,.2f} | R=${res:,.2f}" if sup and res else (f": S=${sup:,.2f}" if sup else f": R=${res:,.2f}")
            table.add_row("S/R Levels", Text(sr_text, style="yellow"))

        # Routing & Trailing Mode
        routing = self.position_data.get('execution_routing') or self.execution_data.get('execution_routing', 'HYBRID_OPTIMIZED')
        if "LIMIT" in routing:
            table.add_row("Routing", Text(": LIMIT (Maker 0.02% Fee)", style="bold green"))
        else:
            table.add_row("Routing", Text(": MARKET (Momentum Breakout)", style="bold cyan"))
        table.add_row("Trailing Mode", ": Stepped Margin + 1M Structural Swing")

        # Position health
        health = self.position_data.get('health', '--')
        if health and health != "--":
            h_color = {"STRONG": "green", "WEAK": "yellow", "EXIT": "red"}.get(health, "white")
            table.add_row("Health", Text(f": {health}", style=h_color))
        
        return Panel(table, title="POSITION", border_style="magenta")


    def _build_signal_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")

        # Live feed prices if available
        btc_p = self.signal_data.get('btc_price', 0.0)
        eth_p = self.signal_data.get('eth_price', 0.0)
        if btc_p and eth_p:
            table.add_row("Live Feeds", Text(f": BTC ${btc_p:,.2f} | ETH ${eth_p:,.2f}", style="bold green"))
        elif btc_p:
            table.add_row("Live BTC", Text(f": ${btc_p:,.2f}", style="bold green"))

        table.add_row("Rankings", Text(f": {self.signal_data.get('rankings', '--')}", style="bold cyan"))
        table.add_row("15M Bias", f": {self.signal_data.get('15m_bias', '--')}")
        table.add_row("5M Struct", f": {self.signal_data.get('5m_bos_choch', '--')}")
        table.add_row("Liq Sweep", f": {self.signal_data.get('liquidity_sweep', '--')}")
        table.add_row("1M Displace", f": {self.signal_data.get('1m_displacement', '--')}")
        table.add_row("Retest", f": {self.signal_data.get('retest', '--')}")
        table.add_row("VWAP / RSI", f": {self.signal_data.get('vwap', '--')} | RSI {self.signal_data.get('rsi', '--')}")
        table.add_row("Volume", f": {self.signal_data.get('volume', '--')}")
        table.add_row("ONNX Conf", f": {self.signal_data.get('onnx_confidence', '--')}")
        table.add_row("Pattern Memory", Text(f": {self.signal_data.get('pattern_memory_stats', 'W: 0 | L: 0')}", style="bold green"))
        table.add_row("Pattern Audit", Text(f": {self.signal_data.get('last_pattern_audit', 'Neutral')}", style="cyan"))
        table.add_row("LLM Status", f": {self.signal_data.get('llm_status', 'Disabled')}")
        table.add_row("Score", f": {self.signal_data.get('signal_score', '--')}")

        setup_t = self.signal_data.get('setup_type', 'NONE')
        if setup_t and setup_t != "NONE":
            table.add_row("Scalp Setup", Text(f": {setup_t}", style="bold magenta"))

        pat = self.signal_data.get('pattern', 'NONE')
        if pat and pat != "NONE":
            table.add_row("1M Pattern", Text(f": {pat}", style="bold yellow"))

        trigger = self.signal_data.get('next_trigger')
        if trigger:
            col = "green" if "READY" in trigger.upper() else "yellow"
            table.add_row("Next Trigger", Text(f": {trigger}", style=col))

        return Panel(table, title="SIGNAL", border_style="yellow")

    def _build_risk_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        risk_check = self.risk_data.get("risk_check", "PASS")
        color = "green" if risk_check == "PASS" else "red"
        table.add_row("Risk Check", Text(f": {risk_check}", style=color))
        
        table.add_row("Max Loss", f": {self.risk_data.get('max_loss', '--')}")
        table.add_row("Target TP", f": {self.risk_data.get('target_profit', '--')}")
        table.add_row("Time Limit", f": {self.risk_data.get('time_limit', '29m (Delta Scalper)')}")
        table.add_row("Spread", f": {self.risk_data.get('spread', '--')}")
        table.add_row("Slippage", f": {self.risk_data.get('slippage', '--')}")
        table.add_row("Liq Dist", f": {self.risk_data.get('liquidation_dist', '--')}")
        table.add_row("Status", f": {self.risk_data.get('risk_status', 'READY')}")
        
        return Panel(table, title="RISK", border_style="red")

    def _build_execution_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        status_raw = str(self.execution_data.get('order_status', '--')).upper()
        if "FILLED" in status_raw:
            status_text = Text(f": {status_raw}", style="bold green")
            border_col = "green"
        elif "OPEN" in status_raw or "PENDING" in status_raw:
            status_text = Text(f": {status_raw}", style="bold yellow")
            border_col = "yellow"
        elif "CLOSED" in status_raw:
            status_text = Text(f": {status_raw}", style="bold cyan")
            border_col = "cyan"
        elif "IDLE" in status_raw:
            status_text = Text(f": {status_raw}", style="dim cyan")
            border_col = "blue"
        else:
            status_text = Text(f": {status_raw}", style="white")
            border_col = "cyan"

        last_ev = str(self.execution_data.get('last_event', '--'))
        if "+" in last_ev or "PROFIT" in last_ev or "TP_HIT" in last_ev:
            event_text = Text(f": {last_ev}", style="bold green")
        elif "-" in last_ev or "LOSS" in last_ev or "SL_HIT" in last_ev:
            event_text = Text(f": {last_ev}", style="bold red")
        else:
            event_text = Text(f": {last_ev}", style="white")

        table.add_row("Order ID", f": {self.execution_data.get('order_id', '--')}")
        table.add_row("Status", status_text)
        table.add_row("Fill Price", f": {self.execution_data.get('fill_price', '--')}")
        table.add_row("Fees", f": {self.execution_data.get('fees', '--')}")
        routing = self.execution_data.get('execution_routing', 'HYBRID_OPTIMIZED')
        table.add_row("Routing Mode", Text(f": {routing}", style="bold cyan"))
        table.add_row("Last Event", event_text)
        
        return Panel(table, title="EXECUTION", border_style=border_col)

    def render(self) -> Layout:
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        
        layout["body"].split_column(
            Layout(name="row1", ratio=1),
            Layout(name="row2", ratio=1)
        )
        
        has_active_pos = bool(
            self.position_data
            and self.position_data.get('symbol')
            and self.position_data.get('symbol') != '--'
        )

        if has_active_pos:
            layout["row1"].split_row(
                Layout(name="account", ratio=1),
                Layout(name="position", ratio=1),
                Layout(name="market_watch", ratio=1),
            )
            layout["position"].update(self._build_position_panel())
        else:
            layout["row1"].split_row(
                Layout(name="account", ratio=1),
                Layout(name="market_watch", ratio=2),
            )

        layout["account"].update(self._build_account_panel())
        layout["market_watch"].update(self._build_market_watch_panel())
        
        layout["row2"].split_row(
            Layout(name="signal", ratio=1),
            Layout(name="risk", ratio=1),
            Layout(name="execution", ratio=1)
        )
        
        mode_color = "green" if self.mode == "PAPER" else "red"
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        header_text = Text(f"[{self.mode}] TRADING BOT - DELTA INDIA  |  UTC: {now_utc}  |  STATUS: RUNNING ({self.target_leverage}x SCALP)", style=f"bold {mode_color}", justify="center")
        layout["header"].update(Panel(header_text))
        
        layout["signal"].update(self._build_signal_panel())
        layout["risk"].update(self._build_risk_panel())
        layout["execution"].update(self._build_execution_panel())
        
        return layout


    async def run(self, update_callback: Callable[[], Awaitable[None]]) -> None:
        logger.info("Starting dashboard loop")
        with Live(self.render(), console=self.console, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    await update_callback()
                    live.update(self.render())
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Dashboard loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in dashboard loop: {e}")
                    await asyncio.sleep(1)
