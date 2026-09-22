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
from rich import box

from src.utils.logger import get_logger

logger = get_logger(__name__)


def render_meter(pct: float, width: int = 10, fill_color: str = "bright_green", empty_color: str = "grey37") -> Text:
    """Render a text progress meter using block characters."""
    clamped = max(0.0, min(100.0, float(pct)))
    filled_len = int(round((clamped / 100.0) * width))
    empty_len = max(0, width - filled_len)
    text = Text()
    text.append("█" * filled_len, style=fill_color)
    text.append("░" * empty_len, style=empty_color)
    return text


def render_badge(label: str, bg_color: str, fg_color: str = "bold black") -> Text:
    """Render a pill-style badge."""
    return Text(f" {label} ", style=f"{fg_color} on {bg_color}")


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

    def _build_header(self) -> Panel:
        header_grid = Table.grid(expand=True)
        header_grid.add_column(justify="left", ratio=1)
        header_grid.add_column(justify="center", ratio=1)
        header_grid.add_column(justify="right", ratio=1)

        # Left: Brand + Mode badge
        mode_bg = "green" if self.mode == "PAPER" else "red"
        mode_badge = render_badge(self.mode, mode_bg, "bold white")
        left_text = Text.assemble(
            ("▲ DELTA SCALPER AI ", "bold bright_cyan"),
            mode_badge
        )

        # Center: Leverage & Strategic Rules
        center_text = Text.assemble(
            (f"{self.target_leverage}x Isolated", "bold white"),
            (" | ", "grey50"),
            ("Jev Sys1 Active", "bold magenta"),
            (" | ", "grey50"),
            ("R:R ≥ 1:1.5R", "bold green"),
        )

        # Right: UTC Clock & Status
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        right_text = Text.assemble(
            ("● RUNNING", "bold bright_green"),
            (" | ", "grey50"),
            (f"{now_utc}", "cyan")
        )

        header_grid.add_row(left_text, center_text, right_text)
        return Panel(header_grid, box=box.ROUNDED, border_style="bright_blue")

    def _build_account_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        net_equity = self.account_data.get("net_equity")
        if net_equity is None:
            net_equity = self.account_data.get("equity", 0.0)
        
        wallet_balance = self.account_data.get("wallet_balance")
        if wallet_balance is None:
            wallet_balance = self.account_data.get("equity", 0.0)
            
        unrealized_pnl = self.account_data.get("unrealized_pnl", 0.0)
        available_margin = self.account_data.get("available_margin", 0.0)
        used_margin = self.account_data.get("used_margin", 0.0)
        used_margin_pct = self.account_data.get("used_margin_pct", 0.0)
        reserve_pct = self.account_data.get("reserve_pct", 20.0)
        daily_pnl = self.account_data.get("daily_pnl", 0.0)
        daily_pnl_pct = self.account_data.get("daily_pnl_pct", 0.0)
        daily_loss_limit = self.account_data.get("daily_loss_limit", -300.0)
        daily_dd_pct = self.account_data.get("daily_drawdown_pct", 0.0)
        win_rate = self.account_data.get("win_rate_pct", 0.0)
        w_trades = self.account_data.get("winning_trades", 0)
        l_trades = self.account_data.get("losing_trades", 0)

        # Net Equity
        eq_style = "bold bright_green" if net_equity >= wallet_balance else "bold bright_white"
        table.add_row("Net Equity", Text(f": ${net_equity:,.2f}", style=eq_style))

        # Wallet Balance
        table.add_row("Wallet Balance", f": ${wallet_balance:,.2f}")

        # Floating uPnL
        upnl_color = "bright_green" if unrealized_pnl > 0 else ("bright_red" if unrealized_pnl < 0 else "grey70")
        upnl_sign = "+" if unrealized_pnl > 0 else ""
        upnl_text = Text(f": {upnl_sign}${unrealized_pnl:,.2f}", style=upnl_color)
        table.add_row("Floating uPnL", upnl_text)

        # Margin Gauge
        margin_meter = render_meter(used_margin_pct, width=8, fill_color="bright_yellow" if used_margin_pct < 50 else "bright_red")
        table.add_row(
            "Margin Used",
            Text.assemble(
                (": ", "white"),
                ("[", "grey50"),
                margin_meter,
                (f"] {used_margin_pct:.1f}%", "bold yellow" if used_margin_pct > 0 else "grey70"),
                (f" (${used_margin:,.2f})", "dim white"),
            )
        )

        # Available Margin & Reserve
        table.add_row("Available Margin", f": ${available_margin:,.2f} ({reserve_pct:.0f}% Res)")

        # Daily P&L
        dpnl_color = "bold bright_green" if daily_pnl >= 0 else "bold bright_red"
        dpnl_sign = "+" if daily_pnl >= 0 else ""
        table.add_row(
            "Today's P&L",
            Text(f": {dpnl_sign}${daily_pnl:,.2f} ({dpnl_sign}{daily_pnl_pct:.2f}%)", style=dpnl_color)
        )

        # Win Rate
        tot = w_trades + l_trades
        wr_color = "bright_green" if win_rate >= 50 else ("yellow" if tot > 0 else "grey70")
        table.add_row(
            "Win Rate",
            Text(f": {win_rate:.1f}% ({w_trades}W - {l_trades}L)", style=wr_color)
        )

        # Daily Loss Shield
        dd_fill_col = "bright_green" if daily_dd_pct < 50 else ("bright_yellow" if daily_dd_pct < 80 else "bold bright_red")
        shield_meter = render_meter(daily_dd_pct, width=8, fill_color=dd_fill_col)
        table.add_row(
            "Risk Shield",
            Text.assemble(
                (": ", "white"),
                ("[", "grey50"),
                shield_meter,
                (f"] -${abs(daily_loss_limit):,.0f} Max ({daily_dd_pct:.1f}% used)", "dim white"),
            )
        )

        return Panel(table, title="ACCOUNT", border_style="blue", box=box.ROUNDED)

    def _build_market_watch_panel(self) -> Panel:
        content_table = Table.grid(padding=(0, 0))
        content_table.add_column(justify="left")

        # Multi-asset radar comparison table
        radar_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            expand=True,
        )
        radar_table.add_column("Asset", style="bold white", width=8)
        radar_table.add_column("Price", justify="right", width=11)
        radar_table.add_column("15M Bias", justify="center", width=9)
        radar_table.add_column("5M Struct", justify="center", width=9)
        radar_table.add_column("Order Block", justify="center", width=15)
        radar_table.add_column("Value Area", justify="center", width=15)
        radar_table.add_column("RSI/Vol", justify="center", width=11)
        radar_table.add_column("Status", justify="center", width=12)

        assets = self.market_watch_data.get("assets", {})
        tracked_symbols = ["BTCUSD", "ETHUSD"]
        for sym in tracked_symbols:
            data = assets.get(sym, {})
            price_val = data.get("delta_price") or data.get("price") or (
                self.signal_data.get("btc_price") if sym == "BTCUSD" else self.signal_data.get("eth_price")
            )
            price_str = f"${price_val:,.2f}" if (isinstance(price_val, (int, float)) and price_val > 0) else "--"
            
            # 15M Bias badge
            bias = str(data.get("bias_15m") or "--").upper()
            if bias == "BULLISH":
                bias_badge = render_badge("BULL", "green", "bold black")
            elif bias == "BEARISH":
                bias_badge = render_badge("BEAR", "red", "bold white")
            else:
                bias_badge = Text(bias, style="grey70")

            # 5M Struct
            bos = data.get("bos_5m", False)
            choch = data.get("choch_5m", False)
            if choch:
                struct_text = Text("CHoCH", style="bold magenta")
            elif bos:
                struct_text = Text("BOS ✓", style="bold green")
            else:
                struct_text = Text("Ranging", style="grey50")

            # Order Block
            if data.get("ob_detected"):
                ob_d = "BULL" if data.get("ob_direction") == "BULLISH" else "BEAR"
                ob_col = "green" if ob_d == "BULL" else "red"
                ob_b = data.get("ob_bottom", 0.0)
                ob_t = data.get("ob_top", 0.0)
                ob_text = Text(f"{ob_d} {ob_b:.0f}-{ob_t:.0f}", style=ob_col)
            else:
                ob_text = Text("--", style="grey50")

            # Value Area
            vah = data.get("vah", 0.0)
            val = data.get("val", 0.0)
            if vah and val:
                va_text = Text(f"{val:.0f}-{vah:.0f}", style="cyan")
            else:
                va_text = Text("--", style="grey50")

            # RSI & Vol
            rsi = data.get("rsi")
            vol = data.get("volume")
            rsi_str = f"{rsi:.0f}" if isinstance(rsi, (int, float)) else "--"
            vol_str = f"{vol:.1f}x" if isinstance(vol, (int, float)) else "--"
            rsi_vol_text = Text(f"{rsi_str} | {vol_str}", style="white")

            # Squeeze / Setup Status
            if data.get("squeeze_fired"):
                status_text = Text("FIRED! ⚡", style="bold bright_green")
            elif data.get("is_squeeze"):
                status_text = Text("SQUEEZE 🔒", style="bold yellow")
            else:
                setup = data.get("setup_type", "NONE")
                if setup and setup != "NONE":
                    status_text = Text(setup[:12], style="bold cyan")
                else:
                    actionable = data.get("actionable", False)
                    status_text = Text("READY" if actionable else "MONITOR", style="green" if actionable else "grey50")

            radar_table.add_row(
                sym,
                price_str,
                bias_badge,
                struct_text,
                ob_text,
                va_text,
                rsi_vol_text,
                status_text
            )

        content_table.add_row(radar_table)

        # Footnote / Strategic status table
        footer_grid = Table.grid(padding=(0, 1))
        footer_grid.add_column(style="dim cyan", justify="left")
        footer_grid.add_column(style="white", justify="left")

        # Level highlights if available
        btc_data = assets.get("BTCUSD", {})
        hl_parts = []
        if btc_data.get("pdh") or btc_data.get("pdl"):
            hl_parts.append(f"PDH ${btc_data.get('pdh', 0):,.0f} | PDL ${btc_data.get('pdl', 0):,.0f}")
        if btc_data.get("poc"):
            hl_parts.append(f"POC ${btc_data.get('poc', 0):,.0f}")
        if btc_data.get("asian_high") or btc_data.get("asian_low"):
            hl_parts.append(f"Asian ${btc_data.get('asian_low', 0):,.0f}-${btc_data.get('asian_high', 0):,.0f}")
        if hl_parts:
            footer_grid.add_row("Key Levels", Text(": " + " | ".join(hl_parts), style="magenta"))

        footer_grid.add_row("Execution", f": {self.target_leverage}x Isolated | Maker 0.02% Fee | 29m Delta Limit Offer")
        
        trigger = self.market_watch_data.get("next_trigger") or self.signal_data.get("next_trigger", "Awaiting 5M BOS/CHoCH + ML >= 65%")
        trigger_col = "bold green" if "READY" in trigger.upper() or "CONFLUENCE" in trigger.upper() else "yellow"
        footer_grid.add_row("Trigger", Text(f": {trigger}", style=trigger_col))

        content_table.add_row(footer_grid)

        return Panel(content_table, title="REAL-TIME MARKET WATCH", border_style="cyan", box=box.ROUNDED)

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
        side = str(self.position_data.get('side', '--')).upper()
        lev = self.position_data.get('leverage')
        lev_val = int(lev) if (isinstance(lev, (int, float)) and lev > 1) else self.target_leverage

        side_badge = render_badge(f"{side} {sym} {lev_val}x", "green" if side in ("BUY", "LONG") else "red", "bold white")
        
        pnl = self.position_data.get("unrealized_pnl", "--")
        margin_pnl_pct = self.position_data.get('margin_pnl_pct', '--')
        
        # PnL Banner
        if isinstance(pnl, (int, float)):
            pnl_sign = "+" if pnl >= 0 else ""
            pnl_col = "bold bright_green" if pnl >= 0 else "bold bright_red"
            pct_str = f" ({margin_pnl_pct:+.2f}%)" if isinstance(margin_pnl_pct, (int, float)) else ""
            pnl_banner = Text(f"{pnl_sign}${pnl:,.2f}{pct_str}", style=pnl_col)
        else:
            pnl_banner = Text(f"{pnl}", style="white")

        table.add_row("Position", Text.assemble(side_badge, "  ", pnl_banner))

        entry = self.position_data.get('entry', '--')
        entry_val = entry if isinstance(entry, (int, float)) else 0.0
        cur = self.position_data.get('current_price', '--')
        cur_val = cur if isinstance(cur, (int, float)) else entry_val
        
        entry_str = f"${entry_val:,.2f}" if entry_val > 0 else f"{entry}"
        cur_str = f"${cur_val:,.2f}" if cur_val > 0 else f"{cur}"
        table.add_row("Price", f": {entry_str} -> {cur_str}")
        
        margin = self.position_data.get('margin', '--')
        margin_val = margin if isinstance(margin, (int, float)) else 0.0
        notional = self.position_data.get('notional', 0.0)
        notional_val = notional if isinstance(notional, (int, float)) else 0.0
        if notional_val <= 0.0 and margin_val > 0:
            notional_val = margin_val * lev_val
        margin_str = f"${margin_val:,.2f}" if margin_val > 0 else f"{margin}"
        notional_str = f"${notional_val:,.2f}" if notional_val > 0 else "--"
        table.add_row("Margin / Notl", f": {margin_str} / {notional_str}")
        
        sl = self.position_data.get('sl', 0.0)
        sl_val = sl if isinstance(sl, (int, float)) else 0.0
        if sl_val <= 0.0 and entry_val > 0:
            is_long = side in ('BUY', 'LONG')
            sl_val = entry_val * (1.0 - 0.03 / lev_val) if is_long else entry_val * (1.0 + 0.03 / lev_val)
        
        sl_str = f"${sl_val:,.2f}" if sl_val > 0 else "--"
        if sl_val > 0 and entry_val > 0:
            is_long = side in ('BUY', 'LONG')
            if (is_long and sl_val > entry_val) or (not is_long and sl_val < entry_val):
                table.add_row("SL (Trailed)", Text.assemble((": " + sl_str + " ", "white"), render_badge("PROFIT LOCKED", "green", "bold black")))
            else:
                table.add_row("SL", f": {sl_str}")
        else:
            table.add_row("SL", f": {sl_str}")
        
        tp = self.position_data.get('tp', '--')
        table.add_row("TP", f": {tp} (Dynamic Trailing)" if tp in ('--', 0, 0.0, None) else (f": ${tp:,.2f}" if isinstance(tp, (int, float)) else f": {tp}"))
        
        # Risk:Reward
        sl_anchor = self.position_data.get('sl_anchor')
        tp_target = self.position_data.get('tp_target_type')
        rr = self.position_data.get('target_rr') or self.position_data.get('realized_rr')
        if rr and isinstance(rr, (int, float)) and rr > 0:
            anchor_lbl = f" ({sl_anchor} -> {tp_target})" if (sl_anchor and tp_target and sl_anchor != 'STATIC') else ""
            table.add_row("R:R Ratio", Text(f": 1:{rr:.2f}R{anchor_lbl}", style="bold green"))

        ht_str = str(self.position_data.get('holding_time', '--'))
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
        table.add_row("Trailing Mode", Text(": Jev AI Sys1 (Ratchet Invariant)", style="bold cyan"))

        # Health
        health = self.position_data.get('health', '--')
        if health and health != "--":
            h_bg = {"STRONG": "green", "WEAK": "yellow", "EXIT": "red"}.get(health, "grey50")
            table.add_row("Health", Text.assemble((": ", "white"), render_badge(str(health), h_bg, "bold black")))
        
        return Panel(table, title="POSITION", border_style="magenta", box=box.ROUNDED)

    def _build_signal_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")

        # Live feed prices if available
        btc_p = self.signal_data.get('btc_price', 0.0)
        eth_p = self.signal_data.get('eth_price', 0.0)
        if btc_p and eth_p:
            table.add_row("Live Feeds", Text(f": BTC ${btc_p:,.2f} | ETH ${eth_p:,.2f}", style="bold bright_green"))
        elif btc_p:
            table.add_row("Live BTC", Text(f": ${btc_p:,.2f}", style="bold bright_green"))

        table.add_row("Rankings", Text(f": {self.signal_data.get('rankings', '--')}", style="bold cyan"))
        table.add_row("15M Bias", f": {self.signal_data.get('15m_bias', '--')}")
        table.add_row("5M Struct", f": {self.signal_data.get('5m_bos_choch', '--')}")
        table.add_row("Liq Sweep", f": {self.signal_data.get('liquidity_sweep', '--')}")
        table.add_row("1M Displace", f": {self.signal_data.get('1m_displacement', '--')}")
        table.add_row("Retest", f": {self.signal_data.get('retest', '--')}")
        table.add_row("VWAP / RSI", f": {self.signal_data.get('vwap', '--')} | RSI {self.signal_data.get('rsi', '--')}")
        table.add_row("Volume", f": {self.signal_data.get('volume', '--')}")
        table.add_row("ONNX Conf", f": {self.signal_data.get('onnx_confidence', '--')}")
        table.add_row("Pattern Mem", Text(f": {self.signal_data.get('pattern_memory_stats', 'W: 0 | L: 0')}", style="bold green"))
        table.add_row("Pattern Audit", Text(f": {self.signal_data.get('last_pattern_audit', 'Neutral')}", style="cyan"))
        
        # Jev AI System 1
        jev_status = self.signal_data.get('jev_status', 'DISABLED')
        jev_bg = "bright_magenta" if "ACTIVE" in str(jev_status).upper() else "grey37"
        table.add_row("Jev AI (Sys1)", Text.assemble((": ", "white"), render_badge(str(jev_status), jev_bg, "bold white")))
        
        jev_vrd = self.signal_data.get('jev_verdict', '--')
        vrd_str = str(jev_vrd)
        vrd_col = "bold green" if "BOOST" in vrd_str else ("bold red" if "VETO" in vrd_str else "cyan")
        table.add_row("Jev Verdict", Text(f": {vrd_str}", style=vrd_col))

        regime = self.signal_data.get('regime_status', 'EXPANSION')
        conv_mult = self.signal_data.get('conviction_mult', '1.00x')
        reg_col = "bold green" if "EXPANSION" in str(regime) else ("bold yellow" if "COMPRESSION" in str(regime) or "RANGING" in str(regime) else "cyan")
        table.add_row("Regime / Size", Text(f": {regime} ({conv_mult})", style=reg_col))

        smt = self.signal_data.get('smt_status', 'SYNC')
        smt_col = "bold bright_red" if "TRAP" in str(smt) or "DIVERGENCE" in str(smt) else "bold green"
        table.add_row("Cross SMT", Text(f": {smt}", style=smt_col))
        
        table.add_row("Score", f": {self.signal_data.get('signal_score', '--')}")

        setup_t = self.signal_data.get('setup_type', 'NONE')
        if setup_t and setup_t != "NONE":
            table.add_row("Scalp Setup", Text(f": {setup_t}", style="bold magenta"))

        pat = self.signal_data.get('pattern', 'NONE')
        if pat and pat != "NONE":
            table.add_row("1M Pattern", Text(f": {pat}", style="bold yellow"))

        trigger = self.signal_data.get('next_trigger')
        if trigger:
            col = "bold green" if "READY" in trigger.upper() or "CONFLUENCE" in trigger.upper() else "yellow"
            table.add_row("Next Trigger", Text(f": {trigger}", style=col))

        return Panel(table, title="SIGNAL", border_style="yellow", box=box.ROUNDED)

    def _build_risk_panel(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="cyan", justify="left")
        table.add_column(style="white", justify="left")
        
        risk_check = self.risk_data.get("risk_check", "PASS")
        rc_bg = "green" if risk_check == "PASS" else "red"
        table.add_row("Risk Check", Text.assemble((": ", "white"), render_badge(str(risk_check), rc_bg, "bold white")))
        
        table.add_row("Max Loss", f": {self.risk_data.get('max_loss', '--')}")
        table.add_row("Target TP", f": {self.risk_data.get('target_profit', '--')}")
        table.add_row("Time Limit", f": {self.risk_data.get('time_limit', '29m (Delta Scalper)')}")
        table.add_row("Spread", f": {self.risk_data.get('spread', '--')}")
        table.add_row("Slippage", f": {self.risk_data.get('slippage', '--')}")
        table.add_row("Liq Dist", f": {self.risk_data.get('liquidation_dist', '--')}")
        table.add_row("Status", f": {self.risk_data.get('risk_status', 'READY')}")
        
        return Panel(table, title="RISK", border_style="red", box=box.ROUNDED)

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
            event_text = Text(f": {last_ev}", style="bold bright_green")
        elif "-" in last_ev or "LOSS" in last_ev or "SL_HIT" in last_ev:
            event_text = Text(f": {last_ev}", style="bold bright_red")
        else:
            event_text = Text(f": {last_ev}", style="white")

        table.add_row("Order ID", f": {self.execution_data.get('order_id', '--')}")
        table.add_row("Status", status_text)
        table.add_row("Fill Price", f": {self.execution_data.get('fill_price', '--')}")
        table.add_row("Fees", f": {self.execution_data.get('fees', '--')}")
        routing = self.execution_data.get('execution_routing')
        if not routing or routing == "HYBRID_OPTIMIZED":
            routing = self.signal_data.get('routing_mode') or "HYBRID_OPTIMIZED"
        table.add_row("Routing Mode", Text(f": {routing}", style="bold cyan"))
        table.add_row("Last Event", event_text)
        
        return Panel(table, title="EXECUTION", border_style=border_col, box=box.ROUNDED)

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
        
        layout["header"].update(self._build_header())
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
