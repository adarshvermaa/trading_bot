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
    def __init__(self, mode: str):
        self.mode = mode.upper()
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
        btc_binance = btc.get("price") or self.signal_data.get("btc_price", 0.0)
        btc_delta_str = f"${btc_delta:,.2f}" if (isinstance(btc_delta, (int, float)) and btc_delta > 0) else "--"
        btc_bias = btc.get("bias_15m") or "--"
        btc_bias_col = "green" if btc_bias == "BULLISH" else ("red" if btc_bias == "BEARISH" else "yellow")
        if isinstance(btc_delta, (int, float)) and isinstance(btc_binance, (int, float)) and btc_delta > 0 and btc_binance > 0 and abs(btc_delta - btc_binance) > 0.01:
            table.add_row("BTC/USD Delta", Text(f": {btc_delta_str}", style="bold green") + Text(f" (Binance: ${btc_binance:,.2f})", style="dim white"))
        else:
            table.add_row("BTC/USD Live", Text(f": {btc_delta_str}", style="bold green"))

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
        eth_binance = eth.get("price") or self.signal_data.get("eth_price", 0.0)
        eth_delta_str = f"${eth_delta:,.2f}" if (isinstance(eth_delta, (int, float)) and eth_delta > 0) else "--"
        eth_bias = eth.get("bias_15m") or "--"
        eth_bias_col = "green" if eth_bias == "BULLISH" else ("red" if eth_bias == "BEARISH" else "yellow")
        if isinstance(eth_delta, (int, float)) and isinstance(eth_binance, (int, float)) and eth_delta > 0 and eth_binance > 0 and abs(eth_delta - eth_binance) > 0.01:
            table.add_row("ETH/USD Delta", Text(f": {eth_delta_str}", style="bold green") + Text(f" (Binance: ${eth_binance:,.2f})", style="dim white"))
        else:
            table.add_row("ETH/USD Live", Text(f": {eth_delta_str}", style="bold green"))

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
        table.add_row("Target Lev", ": 150x Isolated")
        table.add_row("Margin Target", ": -3.0% SL | +6.0% TP")

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
        
        table.add_row("Symbol", f": {self.position_data.get('symbol', '--')}")
        table.add_row("Side", f": {self.position_data.get('side', '--')}")
        
        entry = self.position_data.get('entry', '--')
        table.add_row("Entry", f": ${entry:,.2f}" if isinstance(entry, (int, float)) else f": {entry}")
        
        cur = self.position_data.get('current_price', '--')
        table.add_row("Current", f": ${cur:,.2f}" if isinstance(cur, (int, float)) else f": {cur}")
        
        lev = self.position_data.get('leverage')
        table.add_row("Leverage", f": {lev}x" if lev else ": --")
        
        margin = self.position_data.get('margin', '--')
        table.add_row("Margin", f": ${margin:,.2f}" if isinstance(margin, (int, float)) else f": {margin}")
        
        notional = self.position_data.get('notional', '--')
        table.add_row("Notional", f": ${notional:,.2f}" if isinstance(notional, (int, float)) else f": {notional}")
        
        sl = self.position_data.get('sl', '--')
        table.add_row("SL", f": ${sl:,.2f}" if isinstance(sl, (int, float)) else f": {sl}")
        
        tp = self.position_data.get('tp', '--')
        table.add_row("TP", f": ${tp:,.2f}" if isinstance(tp, (int, float)) else f": {tp}")
        
        pnl = self.position_data.get("unrealized_pnl", "--")
        if isinstance(pnl, (int, float)):
            color = "green" if pnl >= 0 else "red"
            table.add_row("Unrealized P&L", Text(f": ${pnl:,.2f}", style=color))
        else:
            table.add_row("Unrealized P&L", f": {pnl}")
            
        margin_pnl_pct = self.position_data.get('margin_pnl_pct', '--')
        if isinstance(margin_pnl_pct, (int, float)):
            pct_color = "green" if margin_pnl_pct >= 0 else "red"
            table.add_row("Margin P&L %", Text(f": {margin_pnl_pct:.2f}%", style=pct_color))
        else:
            table.add_row("Margin P&L %", f": {margin_pnl_pct}")

        table.add_row("Holding Time", f": {self.position_data.get('holding_time', '--')}")

        # Support/Resistance levels
        sup = self.position_data.get('nearest_support', 0)
        res = self.position_data.get('nearest_resistance', 0)
        if sup or res:
            sr_text = f": S=${sup:,.2f} | R=${res:,.2f}" if sup and res else (f": S=${sup:,.2f}" if sup else f": R=${res:,.2f}")
            table.add_row("S/R Levels", Text(sr_text, style="yellow"))

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
        table.add_row("Spread", f": {self.risk_data.get('spread', '--')}")
        table.add_row("Slippage", f": {self.risk_data.get('slippage', '--')}")
        table.add_row("Liq Dist", f": {self.risk_data.get('liquidation_dist', '--')}")
        table.add_row("Status", f": {self.risk_data.get('risk_status', 'READY')}")
        
        return Panel(table, title="RISK", border_style="red")

    def _build_execution_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        table.add_row("Order ID", f": {self.execution_data.get('order_id', '--')}")
        table.add_row("Status", f": {self.execution_data.get('order_status', '--')}")
        table.add_row("Fill Price", f": {self.execution_data.get('fill_price', '--')}")
        table.add_row("Fees", f": {self.execution_data.get('fees', '--')}")
        table.add_row("Last Event", f": {self.execution_data.get('last_event', '--')}")
        
        return Panel(table, title="EXECUTION", border_style="green")

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
        
        layout["row1"].split_row(
            Layout(name="account", ratio=1),
            Layout(name="position", ratio=1)
        )
        
        layout["row2"].split_row(
            Layout(name="signal", ratio=1),
            Layout(name="risk", ratio=1),
            Layout(name="execution", ratio=1)
        )
        
        mode_color = "green" if self.mode == "PAPER" else "red"
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        header_text = Text(f"[{self.mode}] TRADING BOT - DELTA INDIA  |  UTC: {now_utc}  |  STATUS: RUNNING (150x SCALP)", style=f"bold {mode_color}", justify="center")
        layout["header"].update(Panel(header_text))
        
        layout["account"].update(self._build_account_panel())
        layout["position"].update(self._build_position_panel())
        
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
