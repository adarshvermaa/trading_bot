"""Main entry point for the crypto scalping bot.

Usage:
    python -m src.main --mode paper
    python -m src.main --mode live
    python -m src.main --mode scan
    python -m src.main --mode status
    python -m src.main --mode backtest
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from src.cli import parse_args
from src.config import load_config, AppConfig
from src.data.delta_ws import DeltaWSClient, CandleStore, Candle
from src.execution.delta import DeltaExchangeClient, normalize_delta_symbol
from src.execution.order_manager import OrderManager, OrderState
from src.strategy.structure import MarketStructure
from src.strategy.signals import SignalGenerator
from src.strategy.regime import RegimeFilter, evaluate_session, SessionKillZone
from src.ml.onnx_model import ONNXScalperModel
from src.ml.pattern_memory import PatternMemoryStore, MarketPatternFingerprint
from src.llm.advisor import LLMAdvisor
from src.llm.jev_client import JevClient, JevPreTradeAudit, JevActiveTradeAudit, JevMistakeForensics
from src.risk.risk_manager import RiskManager
from src.portfolio.account import AccountManager
from src.ui.dashboard import Dashboard
from src.utils.logger import setup_logging, get_logger, new_correlation_id


logger = get_logger(__name__)


class ScalpingBot:
    """Orchestrates all components of the scalping bot."""

    def __init__(self, config: AppConfig, mode: str, paper_balance: float = 10_000.0):
        self.config = config
        self.mode = mode
        self.is_live = (mode == "live" or (mode == "status" and config.env.live_trading)) and config.env.live_trading
        self._shutdown = asyncio.Event()

        # ---- Data layer (Delta Exchange WebSocket & CandleStore) ----
        target_symbols = (
            list(config.strategy.assets.universe)
            if (config.strategy and getattr(config.strategy, "assets", None) and config.strategy.assets.universe)
            else ["BTCUSD", "ETHUSD"]
        )
        self.delta_ws = DeltaWSClient(
            symbols=target_symbols,
            ws_url=config.env.delta_ws_url,
            rest_url=config.env.delta_api_url,
            stale_data_seconds=config.risk.failsafe.stale_data_seconds,
            max_retries=config.risk.failsafe.reconnect_max_retries,
            backoff_base=config.risk.failsafe.reconnect_backoff_base,
        )
        self.candle_store = self.delta_ws.candle_store
        self.binance_ws = self.delta_ws  # Backward compatibility alias

        # ---- Execution layer ----
        self.delta_client = DeltaExchangeClient(
            api_key=config.env.delta_api_key,
            api_secret=config.env.delta_api_secret,
            base_url=config.env.delta_api_url,
            ws_url=config.env.delta_ws_url,
            live_trading=self.is_live,
            paper_balance=paper_balance,
        )

        # ---- Strategy layer ----
        self.market_structure = MarketStructure(config.strategy.structure)
        self.signal_generator = SignalGenerator(config.strategy.indicators)
        self.regime_filter = RegimeFilter()

        # ---- ML layer ----
        self.onnx_model = ONNXScalperModel(config.strategy.ml.model_path)

        # ---- Pattern Memory Layer ----
        self.pattern_memory = PatternMemoryStore()

        # ---- LLM layer ----
        self.llm_advisor = LLMAdvisor(
            api_key=config.env.litellm_api_key,
            model=config.env.litellm_model,
        )

        # ---- Jev AI (System One) layer ----
        self.jev_client = JevClient(
            api_key=config.env.jev_api_key,
            config=config.strategy.jev,
        )

        # ---- Risk layer ----
        self.risk_manager = RiskManager(config.risk, jev_client=self.jev_client)

        # ---- Portfolio layer ----
        self.account_manager = AccountManager(
            is_paper=not self.is_live,
            paper_balance=paper_balance,
        )

        # ---- Order manager ----
        self.order_manager = OrderManager(
            delta_client=self.delta_client,
            risk_manager=self.risk_manager,
            config=config,
            account_manager=self.account_manager,
            pattern_memory=self.pattern_memory,
            jev_client=self.jev_client,
        )

        # ---- UI layer ----
        target_lev = getattr(getattr(self.config, "risk", None), "leverage", None)
        high_lev = getattr(target_lev, "high_leverage_value", 100) if target_lev else 100
        self.dashboard = Dashboard(mode=mode, target_leverage=high_lev)

        # ---- State ----
        self._last_scan_time: float = 0.0
        self._best_signal: dict[str, Any] | None = None
        self._position_health: str = "--"
        self._position_health_reason: str = ""
        self._nearest_support: float = 0.0
        self._nearest_resistance: float = 0.0
        self._last_close_reason: str = ""
        self._last_pattern_audit: str = "NEUTRAL"
        self._last_jev_verdict: str = "--"
        self._last_regime_check_time: float = 0.0
        self._last_regime_result: Optional[Any] = None
        self._last_smt_result: Optional[Any] = None
        self._last_conviction_mult: float = 1.0
        self._last_routing_choice: str = "MAKER_POST_ONLY"
        self._re_entry_cooldown_until: float = 0.0
        self._asset_rankings: list[dict[str, Any]] = []
        self._market_watch: dict[str, Any] = {"assets": {}}
        self._monitored_market: dict[str, Any] = {
            "rankings": "BOOTSTRAPPING",
            "15m_bias": "BOOTSTRAPPING",
            "5m_bos_choch": "PRE-SEEDING DATA",
            "liquidity_sweep": "--",
            "1m_displacement": "--",
            "retest": "--",
            "vwap": "--",
            "rsi": "--",
            "volume": "--",
            "onnx_confidence": "--",
            "llm_status": self.llm_advisor.get_status(),
            "jev_status": self.jev_client.get_status(),
            "jev_verdict": "--",
            "signal_score": "WARMING UP",
            "next_trigger": "Pre-seeding candles from Delta Exchange...",
        }

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _handle_shutdown(self, *_: Any) -> None:
        logger.info("Shutdown signal received")
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Data feed
    # ------------------------------------------------------------------

    async def _run_delta_ws(self) -> None:
        """Run the Delta Exchange WebSocket feed."""
        try:
            await self.delta_ws.run()
        except asyncio.CancelledError:
            logger.info("Delta WS task cancelled")
        except Exception:
            logger.exception("Delta WS fatal error")

    # Backward compatibility alias
    async def _run_binance(self) -> None:
        await self._run_delta_ws()

    # ------------------------------------------------------------------
    # Live price helper
    # ------------------------------------------------------------------

    def _get_live_price(self, symbol: str, prefer_delta: bool = True) -> float:
        """Get the latest real-time price for a symbol from Delta Exchange."""
        p = self.delta_ws.get_latest_price(symbol)
        if p and p > 0:
            return p
        delta_p = self.delta_client.get_latest_price(symbol)
        if delta_p and delta_p > 0:
            return delta_p
        candles_1m = self.delta_ws.candle_store.get_candles(symbol.upper(), "1m")
        if candles_1m:
            return candles_1m[-1].close
        return 0.0

    # ------------------------------------------------------------------
    # Position monitor loop (1-second real-time P&L + SL/TP + health)
    # ------------------------------------------------------------------

    async def _position_monitor_loop(self) -> None:
        """Monitor active position every 1s: update P&L, check SL/TP, evaluate health."""
        import numpy as np
        logger.info("Position monitor loop started")
        while not self._shutdown.is_set():
            try:
                active = self.order_manager.active_order
                if not active or active.state != OrderState.FILLED:
                    # Watchdog: If an order is in OPEN state, sync status or cancel on timeout
                    if active and active.state == OrderState.OPEN:
                        await self.order_manager.sync_order_status(active.order_id)
                        exec_cfg = getattr(getattr(self.config, "strategy", None), "execution", None)
                        unfilled_timeout = getattr(exec_cfg, "unfilled_timeout_seconds", 10.0) if exec_cfg else 10.0
                        cancelled = await self.order_manager.check_unfilled_timeouts(unfilled_timeout)
                        if cancelled:
                            self._position_health = "--"
                            self._last_close_reason = "UNFILLED_TIMEOUT"
                    await asyncio.sleep(1)
                    continue


                symbol = active.symbol
                product_id = active.product_id
                live_price = self._get_live_price(symbol, prefer_delta=True)

                if live_price <= 0:
                    await asyncio.sleep(1)
                    continue

                # Live trading check: Verify if position still exists on Delta Exchange
                if self.is_live:
                    is_in_account = bool(self.account_manager and symbol in self.account_manager.positions)
                    if not is_in_account or int(time.time()) % 3 == 0:
                        positions = await self.delta_client.get_positions()
                        reconciled = self.order_manager.sync_exchange_positions(positions, current_price=live_price)
                        if reconciled or not self.order_manager.active_order:
                            last_c = self.order_manager.last_closed_order or {}
                            c_reason = last_c.get("reason", "DELTA_CLOSED")
                            c_pnl = last_c.get("pnl", 0.0)
                            logger.info(
                                f"Live position for {symbol} closed on Delta Exchange: "
                                f"reason={c_reason}, pnl=${c_pnl:+.2f}. Transitioning monitor loop to idle."
                            )
                            self._last_close_reason = f"{c_reason} (${c_pnl:+.2f})"
                            self._position_health = "--"
                            cooldown = 5.0 if c_pnl >= 0 else float(self.config.risk.daily_limits.cooldown_seconds)
                            self._re_entry_cooldown_until = time.time() + cooldown
                            continue

                # --- 1. Update real-time P&L ---
                if self.account_manager and symbol in self.account_manager.positions:
                    self.account_manager.update_current_price(symbol, live_price)

                # --- 1a. Check Delta Scalper Offer time limit (29m for BTC/ETH, 14m for others) ---
                trailing_cfg = getattr(self.config.risk, "trailing_stop", None)
                max_holding_sec = getattr(trailing_cfg, "scalper_offer_max_seconds_major", 1740) if symbol in ("BTCUSD", "ETHUSD") else getattr(trailing_cfg, "scalper_offer_max_seconds_other", 840)
                position_age = time.time() - getattr(active, "created_at", time.time())

                # Check if this position is an institutional breakout runner
                breakout_cfg = getattr(getattr(self.config, "strategy", None), "breakout", None)
                is_breakout_runner = (
                    breakout_cfg and getattr(breakout_cfg, "runner_mode_enabled", False)
                    and getattr(active, "setup_type", "") in ("HTF_BREAKOUT", "BREAKOUT_RETEST")
                    and getattr(self, "_current_regime", "RANGING") == "TRENDING"
                )

                if position_age >= max_holding_sec and not is_breakout_runner:
                    logger.info(
                        f"Delta Scalper Offer limit reached for {symbol} ({position_age:.0f}s >= {max_holding_sec}s). "
                        f"Closing position at {live_price:.2f} to guarantee zero closing fee."
                    )
                    product = await self.delta_client.get_product(symbol)
                    cv = float(product.get("contract_value", 1.0)) if product else 1.0

                    pnl = await self.order_manager.close_and_record(
                        reason="DELTA_SCALPER_OFFER_29M_LIMIT",
                        close_price=live_price,
                        contract_value=cv,
                    )
                    self.order_manager.clear_closed_orders()

                    self._last_close_reason = "SCALPER_OFFER_29M"
                    self._position_health = "--"
                    self._re_entry_cooldown_until = time.time() + 5.0
                    continue

                # --- 1b. Dynamic Stepped Trailing Stop Loss on Margin P&L & Breakeven Locking ---
                if product_id:
                    product = await self.delta_client.get_product(symbol)
                    cv = float(product.get("contract_value", 1.0)) if product else 1.0
                    tick_size = float(product.get("tick_size", 0.0)) if product else None
                    pos_entry = self.account_manager.positions.get(symbol) if self.account_manager else None
                    margin = pos_entry.margin if (pos_entry and pos_entry.margin > 0) else (active.entry_price * active.size * cv / max(1, self.config.risk.leverage.high_leverage_value))

                    # 1b-i. Check Breakeven Trigger (lock SL to entry +0.1% buffer when profit >= 2.0R)
                    be_r_mult = getattr(breakout_cfg, "breakeven_r_mult", 2.0) if breakout_cfg else 2.0
                    initial_sl_val = getattr(active, "initial_sl", None) or active.sl_price or 0.0
                    be_sl, be_updated, be_msg = self.risk_manager.check_breakeven_trigger(
                        entry_price=active.entry_price,
                        current_price=live_price,
                        side=active.side,
                        initial_sl=initial_sl_val,
                        current_sl=active.sl_price or 0.0,
                        tick_size=tick_size,
                        r_multiple=be_r_mult,
                    )
                    if be_updated:
                        self.order_manager.update_active_sl(be_sl)
                        await self.delta_client.update_bracket_stop_loss(
                            product_id,
                            be_sl,
                            tick_size,
                            order_id=active.order_id,
                            side=active.side,
                            size=active.size,
                        )
                        logger.info(f"[BREAKEVEN SL] {symbol}: {be_msg}")

                    # 1b-ii. Dynamic Trailing Stop Loss (Jev AI System One with Structural Swing & Stepped Fallback)
                    c_1m = (
                        self.candle_store.get_candles(symbol, "1m")
                        if (hasattr(self, "candle_store") and self.candle_store)
                        else (self.delta_ws.candle_store.get_candles(symbol, "1m") if hasattr(self, "delta_ws") and self.delta_ws else [])
                    )
                    swing_low_1m = None
                    swing_high_1m = None
                    if c_1m and len(c_1m) >= 5:
                        sub_lows = [float(c.low if hasattr(c, "low") else c["low"]) for c in c_1m[-15:]]
                        sub_highs = [float(c.high if hasattr(c, "high") else c["high"]) for c in c_1m[-15:]]
                        swing_low_1m = min(sub_lows[-5:])
                        swing_high_1m = max(sub_highs[-5:])

                    live_tech_levels = {
                        "swing_low": swing_low_1m,
                        "swing_high": swing_high_1m,
                        "rsi": getattr(self, "_monitored_market", {}).get("rsi", 50.0),
                    }
                    try:
                        live_tech_levels["rsi"] = float(live_tech_levels["rsi"])
                    except (ValueError, TypeError):
                        live_tech_levels["rsi"] = 50.0

                    hold_duration = time.time() - (active.created_at or time.time())
                    pos_atr = (
                        getattr(self, "_current_atr", {}).get(symbol, 100.0)
                        if hasattr(self, "_current_atr")
                        else 100.0
                    )

                    new_sl, updated, msg = await self.risk_manager.calculate_dynamic_jev_trailing_stop(
                        symbol=symbol,
                        side=active.side,
                        entry_price=active.entry_price,
                        current_price=live_price,
                        current_sl=active.sl_price or 0.0,
                        initial_sl=initial_sl_val,
                        margin=margin,
                        contract_value=cv,
                        size=int(active.size),
                        duration_seconds=hold_duration,
                        atr=pos_atr,
                        technical_levels=live_tech_levels,
                        tick_size=tick_size,
                    )
                    if updated:
                        self.order_manager.update_active_sl(new_sl)
                        await self.delta_client.update_bracket_stop_loss(
                            product_id,
                            new_sl,
                            tick_size,
                            order_id=active.order_id,
                            side=active.side,
                            size=active.size,
                        )
                        logger.info(f"[DYNAMIC TRAILING SL] {symbol}: {msg}")

                # --- 2. Check SL/TP triggers (Dual-Layer: Live Exchange Execution + Bot Failsafe) ---
                trigger_reason: Optional[str] = None
                if not self.delta_client.live_trading and product_id:
                    trigger_reason = self.delta_client.check_paper_sl_tp(product_id, live_price)
                elif product_id and active.entry_price > 0:
                    # Live trading failsafe: monitor live tick against active SL & TP
                    is_long = active.side.lower() in ("buy", "long")
                    # Check SL
                    if active.sl_price and active.sl_price > 0:
                        if is_long and live_price <= active.sl_price:
                            trigger_reason = "TRAILING_SL_HIT" if active.sl_price > active.entry_price else "SL_HIT"
                        elif not is_long and live_price >= active.sl_price:
                            trigger_reason = "TRAILING_SL_HIT" if active.sl_price < active.entry_price else "SL_HIT"

                    # Check TP
                    if not trigger_reason and active.tp_price and active.tp_price > 0:
                        if is_long and live_price >= active.tp_price:
                            trigger_reason = "TP_HIT"
                        elif not is_long and live_price <= active.tp_price:
                            trigger_reason = "TP_HIT"

                if trigger_reason:
                    reason = trigger_reason
                    logger.info(f"Position trigger {reason} for {symbol} at {live_price:.2f} (SL={active.sl_price}, TP={active.tp_price})")

                    # Get contract_value for P&L calculation
                    product = await self.delta_client.get_product(symbol)
                    cv = float(product.get("contract_value", 1.0)) if product else 1.0

                    pnl = await self.order_manager.close_and_record(
                        reason=reason,
                        close_price=live_price,
                        contract_value=cv,
                    )
                    self.order_manager.clear_closed_orders()

                    self._last_close_reason = reason
                    self._position_health = "--"

                    # Set re-entry cooldown (5s for TP or profit exit, standard for SL)
                    cooldown = 5.0 if (reason == "TP_HIT" or pnl >= 0) else float(self.config.risk.daily_limits.cooldown_seconds)
                    self._re_entry_cooldown_until = time.time() + cooldown

                    logger.info(
                        f"Position {reason}: pnl={pnl:.2f}, re-entry cooldown={cooldown}s"
                    )
                    continue

                # --- 3. Evaluate position health (every 5 seconds) ---
                if int(time.time()) % 5 == 0:
                    candles_1m = self.delta_ws.candle_store.get_candles(symbol, "1m")
                    candles_5m = self.delta_ws.candle_store.get_candles(symbol, "5m")

                    if len(candles_1m) >= 20 and len(candles_5m) >= 3:
                        health, reason = self.signal_generator.evaluate_position_health(
                            position_side=active.side,
                            entry_price=active.entry_price,
                            current_price=live_price,
                            candles_1m=candles_1m,
                            candles_5m=candles_5m,
                            nearest_support=self._nearest_support,
                            nearest_resistance=self._nearest_resistance,
                            position_age_seconds=time.time() - getattr(active, "created_at", time.time()),
                        )
                        self._position_health = health
                        self._position_health_reason = reason

                        # If health is EXIT, close position early
                        if health == "EXIT":
                            logger.warning(
                                f"Position health EXIT for {symbol}: {reason}. Closing early."
                            )
                            product = await self.delta_client.get_product(symbol)
                            cv = float(product.get("contract_value", 1.0)) if product else 1.0

                            pnl = await self.order_manager.close_and_record(
                                reason=f"HEALTH_EXIT: {reason}",
                                close_price=live_price,
                                contract_value=cv,
                            )
                            self.order_manager.clear_closed_orders()

                            self._last_close_reason = f"HEALTH_EXIT"
                            self._position_health = "--"
                            self._re_entry_cooldown_until = time.time() + 10.0
                            continue

                        # Jev AI active trade health check (System One)
                        if (
                            self.config.strategy.jev.enabled
                            and self.config.strategy.jev.enable_active_monitoring
                            and self.jev_client.enabled
                        ):
                            pos_entry = self.account_manager.positions.get(symbol) if self.account_manager else None
                            pos_pnl = pos_entry.unrealized_pnl if pos_entry else 0.0

                            # Real-time technical metrics for position health
                            c_1m = self.delta_ws.candle_store.get_candles(symbol, "1m")
                            cur_rsi = 50.0
                            vwap_dist = 0.0
                            if len(c_1m) >= 15:
                                import numpy as np
                                closes = np.array([c.close for c in c_1m])
                                highs = np.array([c.high for c in c_1m])
                                lows = np.array([c.low for c in c_1m])
                                volumes = np.array([c.volume for c in c_1m])
                                vwap_arr = SignalGenerator.calculate_vwap(highs, lows, closes, volumes)
                                if len(vwap_arr) > 0 and vwap_arr[-1] > 0:
                                    cur_vwap = float(vwap_arr[-1])
                                    vwap_dist = (live_price - cur_vwap) / cur_vwap
                                rsi_arr = SignalGenerator.calculate_rsi(closes, period=14)
                                if len(rsi_arr) > 0:
                                    cur_rsi = float(rsi_arr[-1])

                            jev_active = await self.jev_client.evaluate_active_trade(
                                symbol=symbol,
                                side=active.side,
                                entry_price=active.entry_price,
                                current_price=live_price,
                                pnl=pos_pnl,
                                duration_seconds=time.time() - getattr(active, "created_at", time.time()),
                                rsi=cur_rsi,
                                vwap_dist_pct=vwap_dist,
                                structure_health_notes=reason,
                            )
                            if jev_active and (
                                jev_active.should_exit_early >= 0.80
                                or (jev_active.momentum_state == "REVERSAL" and jev_active.momentum_confidence >= 0.85)
                            ):
                                logger.warning(
                                    f"[JEV ACTIVE EXIT] Triggering emergency early exit for {symbol} {active.side}: "
                                    f"state={jev_active.momentum_state} (exit_prob={jev_active.should_exit_early:.2f})"
                                )
                                product = await self.delta_client.get_product(symbol)
                                cv = float(product.get("contract_value", 1.0)) if product else 1.0
                                pnl = await self.order_manager.close_and_record(
                                    reason=f"JEV_ACTIVE_EXIT: {jev_active.momentum_state}",
                                    close_price=live_price,
                                    contract_value=cv,
                                )
                                self.order_manager.clear_closed_orders()
                                self._last_close_reason = "JEV_ACTIVE_EXIT"
                                self._position_health = "--"
                                self._re_entry_cooldown_until = time.time() + 10.0
                                continue

                            # Jev Pre-Emptive Scratch Exit (Early flat cut before full -3% SL)
                            if hasattr(self.jev_client, "evaluate_scratch_exit"):
                                duration_sec = time.time() - getattr(active, "created_at", time.time())
                                scratch_res = await self.jev_client.evaluate_scratch_exit(
                                    symbol=symbol,
                                    side=active.side,
                                    seconds_held=duration_sec,
                                    unrealized_pnl_pct=(pos_pnl / (active.entry_price * 0.04)) * 100.0 if (active.entry_price > 0 and pos_pnl != 0) else 0.0,
                                    delta_absorbed_str="STALL" if "stalled" in reason.lower() else "NORMAL",
                                    candle_stall_reason=reason,
                                )
                                if scratch_res and scratch_res.should_scratch:
                                    logger.warning(
                                        f"[JEV PRE-EMPTIVE SCRATCH EXIT] Triggering flat cut for {symbol} {active.side}: {scratch_res.reason}"
                                    )
                                    product = await self.delta_client.get_product(symbol)
                                    cv = float(product.get("contract_value", 1.0)) if product else 1.0
                                    pnl = await self.order_manager.close_and_record(
                                        reason=f"SCRATCH_EXIT: {scratch_res.thesis_integrity}",
                                        close_price=live_price,
                                        contract_value=cv,
                                    )
                                    self.order_manager.clear_closed_orders()
                                    self._last_close_reason = "SCRATCH_EXIT"
                                    self._position_health = "--"
                                    self._re_entry_cooldown_until = time.time() + 10.0
                                    continue

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Position monitor error", exc_info=True)

            await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Scanner / strategy loop
    # ------------------------------------------------------------------

    async def _scan_assets(self) -> dict[str, Any] | None:
        """Scan all universe assets (BTC and ETH), rank them, and return the best actionable setup."""
        universe = self.config.strategy.assets.universe
        weights = self.config.strategy.scanner.score_weights

        evaluations: list[dict[str, Any]] = []

        import numpy as np
        def candles_to_arrays(candles: list[Candle]) -> dict[str, np.ndarray]:
            return {
                "open": np.array([c.open for c in candles]),
                "high": np.array([c.high for c in candles]),
                "low": np.array([c.low for c in candles]),
                "close": np.array([c.close for c in candles]),
                "volume": np.array([c.volume for c in candles]),
            }

        from src.strategy.signals import compute_atr, compute_adx, compute_vwap

        # Periodic Jev System One Market Regime Arbiter (every 15 min or first run)
        now_ts = time.time()
        if (
            self.config.strategy.jev.enabled
            and self.jev_client.enabled
            and (now_ts - self._last_regime_check_time >= 900.0 or self._last_regime_result is None)
        ):
            btc_15m = self.delta_ws.candle_store.get_candles("BTCUSD", "15m")
            if len(btc_15m) >= 50:
                try:
                    c_highs = np.array([c.high for c in btc_15m])
                    c_lows = np.array([c.low for c in btc_15m])
                    c_closes = np.array([c.close for c in btc_15m])
                    c_vols = np.array([c.volume for c in btc_15m])
                    atr_15m_arr = compute_atr(c_highs, c_lows, c_closes, 14)
                    atr_15m_val = float(atr_15m_arr[-1]) if len(atr_15m_arr) > 0 else 50.0
                    rel_vol_15m = float(c_vols[-1] / (np.mean(c_vols[-20:]) + 1e-9)) if len(c_vols) >= 20 else 1.0
                    session_info_now = evaluate_session(datetime.now(timezone.utc))
                    regime_res = await self.jev_client.evaluate_market_regime(
                        atr_15m=atr_15m_val,
                        atr_1h=atr_15m_val * 1.5,
                        bb_bandwidth=0.02,
                        rel_vol_15m=rel_vol_15m,
                        session_zone=session_info_now.zone.value if session_info_now else "NORMAL",
                    )
                    self._last_regime_result = regime_res
                    self._last_regime_check_time = now_ts
                except Exception as e:
                    logger.debug(f"Failed to evaluate Jev market regime: {e}")

        for symbol in universe:
            candles_15m = self.delta_ws.candle_store.get_candles(symbol, "15m")
            candles_5m = self.delta_ws.candle_store.get_candles(symbol, "5m")
            candles_1m = self.delta_ws.candle_store.get_candles(symbol, "1m")

            if len(candles_15m) < 50 or len(candles_5m) < 50 or len(candles_1m) < 50:
                continue

            arr_15m = candles_to_arrays(candles_15m)
            arr_5m = candles_to_arrays(candles_5m)
            arr_1m = candles_to_arrays(candles_1m)

            atr_values = compute_atr(arr_1m["high"], arr_1m["low"], arr_1m["close"],
                                     self.config.strategy.indicators.atr_period)
            current_atr = float(atr_values[-1]) if len(atr_values) > 0 else 0.0

            htf_levels = self.delta_ws.get_htf_levels(symbol) if hasattr(self, "delta_ws") and self.delta_ws else None
            try:
                structure = self.market_structure.analyze(
                    arr_15m, arr_5m, arr_1m, current_atr, htf_levels=htf_levels
                )
            except Exception:
                logger.debug(f"Structure analysis failed for {symbol}", exc_info=True)
                continue

            try:
                sig = self.signal_generator.generate(
                    arr_15m, arr_5m, arr_1m, structure
                )
            except Exception:
                logger.debug(f"Signal generation failed for {symbol}", exc_info=True)
                continue

            adx_values = compute_adx(arr_1m["high"], arr_1m["low"], arr_1m["close"],
                                     self.config.strategy.indicators.adx_period)
            current_adx = float(adx_values[-1]) if len(adx_values) > 0 else 0.0
            avg_atr = float(np.mean(atr_values[-20:])) if len(atr_values) >= 20 else current_atr

            regime = self.regime_filter.evaluate(current_adx, current_atr, avg_atr)
            self._current_regime = regime.name

            # Session evaluation & adaptive confidence
            session_cfg = getattr(self.config.strategy, "session", None)
            std_conf = getattr(session_cfg, "standard_min_confidence", 0.65) if session_cfg else 0.65
            dead_conf = getattr(session_cfg, "dead_zone_min_confidence", 0.75) if session_cfg else 0.75
            session_info = evaluate_session(standard_confidence=std_conf, dead_zone_confidence=dead_conf)
            effective_min_conf = session_info.recommended_min_confidence if getattr(session_cfg, "kill_zones_enabled", True) else self.config.strategy.ml.min_confidence

            # Build MarketPatternFingerprint (20 features)
            cur_p = float(arr_1m["close"][-1])
            cur_o = float(arr_1m["open"][-1])
            cur_h = float(arr_1m["high"][-1])
            cur_l = float(arr_1m["low"][-1])
            cur_range = max(cur_h - cur_l, 1e-9)
            u_wick = max(0.0, cur_h - max(cur_o, cur_p)) / cur_range
            l_wick = max(0.0, min(cur_o, cur_p) - cur_l) / cur_range
            b_ratio = abs(cur_p - cur_o) / cur_range

            vah = getattr(structure, "vah", 0.0) or (htf_levels.get("VAH", 0.0) if htf_levels else 0.0)
            val = getattr(structure, "val", 0.0) or (htf_levels.get("VAL", 0.0) if htf_levels else 0.0)
            pdh = getattr(structure, "pdh", 0.0) or (htf_levels.get("PDH", 0.0) if htf_levels else 0.0)
            pdl = getattr(structure, "pdl", 0.0) or (htf_levels.get("PDL", 0.0) if htf_levels else 0.0)

            dist_vah = (cur_p - vah) / vah if vah > 0 else 0.0
            dist_val = (cur_p - val) / val if val > 0 else 0.0
            dist_pdh = (cur_p - pdh) / pdh if pdh > 0 else 0.0
            dist_pdl = (cur_p - pdl) / pdl if pdl > 0 else 0.0

            ema_trend_val = 1.0 if sig.ema_cross == "BULLISH" else (-1.0 if sig.ema_cross == "BEARISH" else 0.0)
            vwap_values = compute_vwap(arr_1m["high"], arr_1m["low"], arr_1m["close"], arr_1m["volume"])
            cur_vwap = float(vwap_values[-1]) if len(vwap_values) > 0 else cur_p
            vwap_dist = (cur_p - cur_vwap) / cur_vwap if cur_vwap > 0 else 0.0

            fingerprint = MarketPatternFingerprint(
                rsi_norm=sig.rsi / 100.0,
                adx_norm=current_adx / 100.0,
                atr_norm=current_atr / cur_p if cur_p > 0 else 0.0,
                vwap_dist=vwap_dist,
                vol_ratio=sig.relative_volume,
                ema_trend=ema_trend_val,
                is_squeeze=1.0 if getattr(structure, "is_squeeze", False) else 0.0,
                structure_score=sig.strength,
                dist_to_vah_pct=dist_vah,
                dist_to_val_pct=dist_val,
                dist_to_pdh_pct=dist_pdh,
                dist_to_pdl_pct=dist_pdl,
                is_bull_trap=1.0 if getattr(structure, "is_bull_trap", False) else 0.0,
                is_bear_trap=1.0 if getattr(structure, "is_bear_trap", False) else 0.0,
                is_judas_swing=1.0 if getattr(structure, "is_judas_swing", False) else 0.0,
                is_volume_absorption=1.0 if getattr(structure, "is_volume_absorption", False) else 0.0,
                upper_wick_ratio=u_wick,
                lower_wick_ratio=l_wick,
                body_ratio=b_ratio,
                spread_bps=0.8,
                symbol=symbol,
                direction=sig.direction if sig.direction != "NONE" else ("LONG" if structure.bias_15m == "BULLISH" else "SHORT"),
                entry_price=cur_p,
            )

            # Pattern Memory Check (Sub-millisecond advisory & confluence)
            pattern_audit = self.pattern_memory.check_pattern(fingerprint)
            if pattern_audit.is_blocked:
                logger.info(
                    f"Pattern Memory ADVISORY for {symbol} {sig.direction}: {pattern_audit.reason} (Delegating judgment to Jev)"
                )
                self._last_pattern_audit = f"ADVISORY ({pattern_audit.matched_loss_trade_id} {pattern_audit.matched_loss_similarity*100:.0f}%)"
                if not (self.config.strategy.jev.enabled and self.jev_client.enabled):
                    # In offline/fallback mode without Jev, apply a mild score dampener rather than hard crash
                    score = max(0.0, score - 0.20)
            elif pattern_audit.is_boosted:
                logger.info(
                    f"Pattern Memory BOOST for {symbol} {sig.direction}: {pattern_audit.reason}"
                )
                self._last_pattern_audit = f"BOOST (+{pattern_audit.confidence_adjustment:.2f})"
                sig.strength = min(1.0, sig.strength + pattern_audit.confidence_adjustment)
            else:
                self._last_pattern_audit = "NEUTRAL"

            # ML confirmation
            ml_confirmed = True
            ml_confidence = 0.0
            if self.config.strategy.ml.enabled and self.onnx_model.is_available:
                features = self.onnx_model.prepare_pattern_features(fingerprint)
                eval_dir = sig.direction if sig.direction != "NONE" else ("LONG" if structure.bias_15m == "BULLISH" else "SHORT")
                ml_confirmed, ml_confidence = self.onnx_model.confirm_signal(
                    eval_dir, features, effective_min_conf
                )

            # Composite score
            rsi_score = (sig.rsi / 100.0) if sig.direction == "LONG" else ((100.0 - sig.rsi) / 100.0 if sig.direction == "SHORT" else 0.5)
            indicator_score = min(max(rsi_score, 0.0), 1.0)
            score = (
                weights.structure * (sig.strength if structure.is_valid else 0.25)
                + weights.regime * (1.0 if regime.name == "TRENDING" else 0.5)
                + weights.indicators * indicator_score
                + weights.ml_confidence * ml_confidence
            )

            # Jev AI (System One) Confluence & Trap Gatekeeper
            jev_audit = None
            if (
                self.config.strategy.jev.enabled
                and self.jev_client.enabled
                and structure.is_valid
                and sig.direction in ("LONG", "SHORT")
            ):
                jev_indicators = {
                    "rsi": sig.rsi,
                    "adx": current_adx,
                    "relative_volume": sig.relative_volume,
                    "vwap_position": getattr(sig, "vwap_position", "NEUTRAL"),
                    "atr": current_atr,
                }
                jev_audit = await self.jev_client.evaluate_pre_trade_setup(
                    symbol=symbol,
                    direction=sig.direction,
                    indicators=jev_indicators,
                    structure=structure,
                    fingerprint=fingerprint,
                    session_info=session_info,
                    pattern_memory_audit=pattern_audit,
                )
                if jev_audit:
                    if jev_audit.is_vetoed:
                        logger.warning(
                            f"[JEV VETO] {symbol} {sig.direction} rejected: {jev_audit.veto_reason}"
                        )
                        if "Poor setup quality" in jev_audit.veto_reason:
                            self._last_jev_verdict = f"VETO (Low Grade {jev_audit.setup_grade:.1f}/4.0 < 1.5)"
                        elif "Direction conflict" in jev_audit.veto_reason:
                            self._last_jev_verdict = f"VETO (Dir conflict vs {jev_audit.direction_bias})"
                        elif "Trap risk" in jev_audit.veto_reason:
                            self._last_jev_verdict = f"VETO (Trap prob {jev_audit.trap_probability:.0%})"
                        elif "Cost friction" in jev_audit.veto_reason:
                            self._last_jev_verdict = "VETO (High Spread/Friction)"
                        elif "Past mistake pattern" in jev_audit.veto_reason:
                            self._last_jev_verdict = f"VETO (Past Mistake Risk {jev_audit.setup_grade:.1f}/4.0)"
                        else:
                            self._last_jev_verdict = f"VETO ({jev_audit.veto_reason[:30]})"
                    elif jev_audit.is_boosted:
                        score += jev_audit.boost_amount
                        self._last_jev_verdict = f"BOOST (+{jev_audit.boost_amount:.2f}, Grd={jev_audit.setup_grade:.1f})"
                    else:
                        self._last_jev_verdict = f"PASS (Grd={jev_audit.setup_grade:.1f})"

            score_hurdle = 0.70 if session_info.is_dead_zone else 0.55
            is_actionable = (
                structure.is_valid
                and sig.direction in ("LONG", "SHORT")
                and not (jev_audit and jev_audit.is_vetoed)
                and (ml_confirmed or score >= score_hurdle)
            )

            # Extract ICT structural levels for Jev dynamic SL/TP
            swing_low_val = min([c.low for c in candles_1m[-15:]]) if len(candles_1m) >= 15 else (candles_1m[-1].low if candles_1m else None)
            swing_high_val = max([c.high for c in candles_1m[-15:]]) if len(candles_1m) >= 15 else (candles_1m[-1].high if candles_1m else None)
            tech_levels = {
                "swing_low": swing_low_val,
                "swing_high": swing_high_val,
                "ob_bottom": structure.ob_bottom if getattr(structure, "ob_bottom", 0.0) > 0 else None,
                "ob_top": structure.ob_top if getattr(structure, "ob_top", 0.0) > 0 else None,
                "trap_wick_price": structure.trap_wick_extreme if getattr(structure, "trap_wick_extreme", 0.0) > 0 else getattr(sig, "trap_wick_price", None),
                "vah": structure.vah if getattr(structure, "vah", 0.0) > 0 else None,
                "val": structure.val if getattr(structure, "val", 0.0) > 0 else None,
                "pdh": structure.pdh if getattr(structure, "pdh", 0.0) > 0 else None,
                "pdl": structure.pdl if getattr(structure, "pdl", 0.0) > 0 else None,
                "nearest_support": structure.nearest_support if getattr(structure, "nearest_support", 0.0) > 0 else None,
                "nearest_resistance": structure.nearest_resistance if getattr(structure, "nearest_resistance", 0.0) > 0 else None,
                "eqh": structure.eqh if getattr(structure, "eqh", 0.0) > 0 else None,
                "eql": structure.eql if getattr(structure, "eql", 0.0) > 0 else None,
                "fvg_top": structure.fvg_top if getattr(structure, "fvg_top", 0.0) > 0 else None,
                "fvg_bottom": structure.fvg_bottom if getattr(structure, "fvg_bottom", 0.0) > 0 else None,
                "chart_pattern_target": structure.chart_pattern_target if getattr(structure, "chart_pattern_target", 0.0) > 0 else None,
                "pricing_zone": getattr(structure, "pricing_zone", "EQUILIBRIUM"),
                "opposing_liquidity": (structure.eqh if sig.direction == "LONG" else structure.eql) if (structure.eqh > 0 or structure.eql > 0) else None,
            }

            evaluations.append({
                "symbol": symbol,
                "signal": sig,
                "structure": structure,
                "regime": regime,
                "session": session_info,
                "ml_confidence": ml_confidence,
                "ml_confirmed": ml_confirmed,
                "score": score,
                "atr": current_atr,
                "is_actionable": is_actionable,
                "fingerprint": fingerprint,
                "technical_levels": tech_levels,
            })

        if not evaluations:
            return None

        # Cross-Asset ICT SMT Divergence check
        if len(evaluations) >= 2 and self.jev_client and getattr(self.jev_client, "enabled", False):
            btc_e = next((x for x in evaluations if x["symbol"] == "BTCUSD"), None)
            eth_e = next((x for x in evaluations if x["symbol"] == "ETHUSD"), None)
            if btc_e and eth_e:
                btc_5m = self.delta_ws.candle_store.get_candles("BTCUSD", "5m")
                eth_5m = self.delta_ws.candle_store.get_candles("ETHUSD", "5m")
                b_delta = ((btc_5m[-1].close - btc_5m[-2].close) / btc_5m[-2].close * 100.0) if len(btc_5m) >= 2 else 0.0
                e_delta = ((eth_5m[-1].close - eth_5m[-2].close) / eth_5m[-2].close * 100.0) if len(eth_5m) >= 2 else 0.0
                try:
                    self._last_smt_result = await self.jev_client.evaluate_cross_asset_smt(
                        btc_delta_5m=b_delta,
                        eth_delta_5m=e_delta,
                        btc_bias=btc_e["structure"].bias_15m,
                        eth_bias=eth_e["structure"].bias_15m,
                        btc_bos=btc_e["structure"].bos_5m,
                        eth_bos=eth_e["structure"].bos_5m,
                    )
                    if self._last_smt_result and self._last_smt_result.is_trap_warning:
                        if self._last_smt_result.favored_asset == "TRADE_BTC":
                            eth_e["is_actionable"] = False
                        elif self._last_smt_result.favored_asset == "TRADE_ETH":
                            btc_e["is_actionable"] = False
                        elif self._last_smt_result.favored_asset == "AVOID_BOTH":
                            btc_e["is_actionable"] = False
                            eth_e["is_actionable"] = False
                except Exception as ex:
                    logger.debug(f"SMT check error: {ex}")

        # Sort rankings: Actionable triggers first, then by highest score
        evaluations.sort(key=lambda x: (1 if x["is_actionable"] else 0, x["score"]), reverse=True)
        self._asset_rankings = evaluations

        top = evaluations[0]
        conviction_mult = 1.0
        routing_choice = "MAKER_POST_ONLY"

        if top["is_actionable"] and self.jev_client and getattr(self.jev_client, "enabled", False):
            # Conviction Sizing (0.5x to 1.5x)
            try:
                grade_val = 3.0 if top.get("ml_confirmed") else 2.5
                size_res = await self.jev_client.evaluate_conviction_sizing(
                    symbol=top["symbol"],
                    setup_grade=grade_val,
                    dir_conf=top.get("score", 0.8),
                    onnx_conf=top.get("ml_confidence", 0.7),
                )
                if size_res:
                    conviction_mult = size_res.conviction_multiplier
                    self._last_conviction_mult = conviction_mult
            except Exception as ex:
                logger.debug(f"Conviction sizing error: {ex}")

            # Smart Execution Routing (Maker vs Market)
            try:
                route_res = await self.jev_client.evaluate_execution_routing(
                    symbol=top["symbol"],
                    side=top["signal"].direction,
                    spread_bps=1.0,
                    tape_velocity_1m=1.5,
                )
                if route_res:
                    routing_choice = route_res.order_type
                    self._last_routing_choice = routing_choice
            except Exception as ex:
                logger.debug(f"Routing error: {ex}")

        top["conviction_multiplier"] = conviction_mult
        top["execution_routing"] = routing_choice

        # Formatted ranking summary string for dashboard: e.g. "#1 BTCUSD: 0.78 (LONG) | #2 ETHUSD: 0.62 (NONE)"
        rankings_str = " | ".join(
            [f"#{i+1} {e['symbol']} ({e['score']:.2f}, {e['signal'].direction})" for i, e in enumerate(evaluations)]
        )

        next_trigger = "CONFLUENCE READY TO EXECUTE" if top["is_actionable"] else "Awaiting 5M BOS/CHoCH + ML >= 65%"

        # Build real-time market watch data for dashboard
        market_watch: dict[str, Any] = {
            "top_symbol": top["symbol"],
            "top_score": top["score"],
            "rankings": rankings_str,
            "next_trigger": next_trigger,
            "assets": {},
        }
        for e in evaluations:
            sym = e["symbol"]
            cur_p = self._get_live_price(sym)
            delta_p = self.delta_client.get_latest_price(sym) or cur_p
            market_watch["assets"][sym] = {
                "symbol": sym,
                "price": cur_p,
                "delta_price": delta_p,
                "bias_15m": e["structure"].bias_15m,
                "bos_5m": e["structure"].bos_5m,
                "choch_5m": e["structure"].choch_5m,
                "support": e["structure"].nearest_support,
                "resistance": e["structure"].nearest_resistance,
                "direction": e["signal"].direction,
                "score": e["score"],
                "rsi": e["signal"].rsi,
                "volume": getattr(e["signal"], "relative_volume", 1.0),
                "actionable": e["is_actionable"],
                "setup_type": getattr(e["structure"], "setup_type", "NONE"),
                "pattern": getattr(e["signal"], "pattern", "NONE"),
                "pdh": getattr(e["structure"], "pdh", 0.0),
                "pdl": getattr(e["structure"], "pdl", 0.0),
                "poc": getattr(e["structure"], "poc", 0.0),
                "vah": getattr(e["structure"], "vah", 0.0),
                "val": getattr(e["structure"], "val", 0.0),
                "eqh": getattr(e["structure"], "eqh", 0.0),
                "eql": getattr(e["structure"], "eql", 0.0),
                "asian_high": getattr(e["structure"], "asian_high", 0.0),
                "asian_low": getattr(e["structure"], "asian_low", 0.0),
                "ob_detected": getattr(e["structure"], "ob_detected", False),
                "ob_direction": getattr(e["structure"], "ob_direction", "NONE"),
                "ob_top": getattr(e["structure"], "ob_top", 0.0),
                "ob_bottom": getattr(e["structure"], "ob_bottom", 0.0),
                "ob_testing": getattr(e["structure"], "ob_testing", False),
                "is_squeeze": getattr(e["structure"], "is_squeeze", False),
                "squeeze_fired": getattr(e["structure"], "squeeze_fired", False),
                "breakout_direction": getattr(e["structure"], "breakout_direction", "NONE"),
            }
        self._market_watch = market_watch

        btc_p = self._get_live_price("BTCUSD")
        eth_p = self._get_live_price("ETHUSD")

        # Update monitored market with top asset details and live rankings
        self._monitored_market = {
            "btc_price": btc_p,
            "eth_price": eth_p,
            "rankings": rankings_str,
            "15m_bias": f"[{top['symbol']}] {top['structure'].bias_15m}",
            "5m_bos_choch": f"BOS={top['structure'].bos_5m} CHoCH={top['structure'].choch_5m}",
            "liquidity_sweep": str(top['structure'].liquidity_sweep_5m),
            "1m_displacement": str(top['structure'].displacement_1m),
            "retest": str(top['structure'].retest_1m),
            "vwap": getattr(top['signal'], "vwap_position", "NEUTRAL"),
            "rsi": f"{top['signal'].rsi:.1f}",
            "volume": f"{top['signal'].relative_volume:.2f}x",
            "onnx_confidence": f"{top['ml_confidence']:.1%}" if top['ml_confidence'] > 0 else "--",
            "pattern_memory_stats": f"W:{len(self.pattern_memory.winning_patterns)} | L:{len(self.pattern_memory.losing_patterns)}",
            "last_pattern_audit": self._last_pattern_audit,
            "llm_status": self.llm_advisor.get_status(),
            "jev_status": self.jev_client.get_status(),
            "jev_verdict": self._last_jev_verdict,
            "signal_score": f"ACTIONABLE ({top['score']:.3f})" if top["is_actionable"] else f"RANK #{1} {top['symbol']} ({top['score']:.2f})",
            "next_trigger": next_trigger,
            "setup_type": getattr(top["structure"], "setup_type", "NONE"),
            "chart_pattern": getattr(top["structure"], "chart_pattern", "NONE"),
            "chart_pattern_direction": getattr(top["structure"], "chart_pattern_direction", "NONE"),
            "pricing_zone": getattr(top["structure"], "pricing_zone", "EQUILIBRIUM"),
            "playbook": getattr(top["structure"], "playbook", "NONE"),
            "pattern": getattr(top["signal"], "pattern", "NONE"),
            "session": getattr(getattr(top.get("session"), "zone", None), "value", "NORMAL"),
            "fvg": f"{top['structure'].fvg_direction} (testing={top['structure'].fvg_testing})" if getattr(top['structure'], "fvg_detected", False) else "--",
            "regime_status": getattr(self._last_regime_result, "market_regime", "EXPANSION") if self._last_regime_result else "EXPANSION",
            "smt_status": "SMT TRAP ALERT" if (self._last_smt_result and self._last_smt_result.is_trap_warning) else "SYNC",
            "conviction_mult": f"{self._last_conviction_mult:.2f}x",
            "routing_mode": self._last_routing_choice,
        }

        # Return best setup if actionable
        if top["is_actionable"]:
            return top

        return None

    async def _strategy_loop(self) -> None:
        """Main strategy loop: wait for closed candles, scan, and trade.
        
        Mode A (no position): Scan BTC + ETH, enter on confluence.
        Mode B (active position): Re-analyze structure, update S/R levels, 
                                   close on invalidation, record P&L.
        After close: Cooldown → re-scan → re-enter.
        """
        import numpy as np
        logger.info("Strategy loop started")
        while not self._shutdown.is_set():
            try:
                # Wait for a new closed candle
                try:
                    await asyncio.wait_for(
                        self.delta_ws.new_candle_event.wait(),
                        timeout=5.0,
                    )
                    self.delta_ws.new_candle_event.clear()
                except asyncio.TimeoutError:
                    pass  # Just re-check conditions

                # Update risk manager connection status
                self.risk_manager.market_data_connected = self.delta_ws.is_connected or (not self.delta_ws.is_stale)
                self.risk_manager.data_stale = self.delta_ws.is_stale

                # Update account from exchange
                try:
                    if self.is_live:
                        balances = await self.delta_client.get_wallet_balances()
                        positions = await self.delta_client.get_positions()
                        self.account_manager.update_from_exchange(balances, positions)
                        if isinstance(positions, list):
                            ao = self.order_manager.active_order
                            cur_p = self._get_live_price(ao.symbol) if (ao and ao.symbol) else 0.0
                            self.order_manager.sync_exchange_positions(positions, current_price=cur_p)

                    self.risk_manager.delta_connected = True
                    self.risk_manager.delta_error_reason = ""
                except Exception as e:
                    self.risk_manager.delta_connected = False
                    err_str = str(e)
                    if "ip_not_whitelisted" in err_str.lower():
                        self.risk_manager.delta_error_reason = "IP_NOT_WHITELISTED"
                        logger.error(
                            "DELTA LIVE API ERROR: Your IP is not whitelisted for this Delta API key. "
                            "Please add your IP in Delta Exchange API settings or create a non-IP-restricted key."
                        )
                    else:
                        self.risk_manager.delta_error_reason = "DELTA_DISCONNECTED"
                        logger.warning(f"Failed to update account from Delta: {e}")

                # Check if we have an active position
                has_position = await self.order_manager.has_active_position()

                if has_position:
                    # ===== MODE B: Active Position — Structure Monitoring =====
                    active = self.order_manager.active_order
                    if active and active.state == OrderState.FILLED:
                        candles_5m = self.delta_ws.candle_store.get_candles(active.symbol, "5m")
                        candles_1m = self.delta_ws.candle_store.get_candles(active.symbol, "1m")
                        candles_15m = self.delta_ws.candle_store.get_candles(active.symbol, "15m")

                        if len(candles_15m) >= 50 and len(candles_5m) >= 50 and len(candles_1m) >= 50:
                            def to_arr(cc: list) -> dict:
                                return {
                                    "open": np.array([c.open for c in cc]),
                                    "high": np.array([c.high for c in cc]),
                                    "low": np.array([c.low for c in cc]),
                                    "close": np.array([c.close for c in cc]),
                                    "volume": np.array([c.volume for c in cc]),
                                }

                            from src.strategy.signals import compute_atr
                            a1m = to_arr(candles_1m)
                            atr_vals = compute_atr(
                                a1m["high"], a1m["low"], a1m["close"],
                                self.config.strategy.indicators.atr_period,
                            )
                            cur_atr = float(atr_vals[-1]) if len(atr_vals) > 0 else 0.0

                            htf_lvl = self.delta_ws.get_htf_levels(active.symbol) if hasattr(self, "delta_ws") and self.delta_ws else None
                            try:
                                struct = self.market_structure.analyze(
                                    to_arr(candles_15m), to_arr(candles_5m), a1m, cur_atr, htf_levels=htf_lvl
                                )
                                # Update S/R levels from live structure analysis
                                self._nearest_support = struct.nearest_support
                                self._nearest_resistance = struct.nearest_resistance

                                # Update signal panel with live analysis while position is open
                                sig = self.signal_generator.generate(
                                    to_arr(candles_15m), to_arr(candles_5m), a1m, struct
                                )
                                btc_live = self._get_live_price("BTCUSD")
                                eth_live = self._get_live_price("ETHUSD")
                                self._monitored_market = {
                                    "btc_price": btc_live,
                                    "eth_price": eth_live,
                                    "rankings": self._monitored_market.get("rankings", "#1 BTCUSD | #2 ETHUSD"),
                                    "15m_bias": f"[{active.symbol}] {struct.bias_15m}",
                                    "5m_bos_choch": f"BOS={struct.bos_5m} CHoCH={struct.choch_5m}",
                                    "liquidity_sweep": str(struct.liquidity_sweep_5m),
                                    "1m_displacement": str(struct.displacement_1m),
                                    "retest": str(struct.retest_1m),
                                    "vwap": getattr(sig, "vwap_position", "--"),
                                    "rsi": f"{sig.rsi:.1f}",
                                    "volume": f"{sig.relative_volume:.2f}x",
                                    "onnx_confidence": self._monitored_market.get("onnx_confidence", "--"),
                                    "pattern_memory_stats": f"W:{len(self.pattern_memory.winning_patterns)} | L:{len(self.pattern_memory.losing_patterns)}",
                                    "last_pattern_audit": self._last_pattern_audit,
                                    "llm_status": self.llm_advisor.get_status(),
                                    "jev_status": self.jev_client.get_status(),
                                    "jev_verdict": self._last_jev_verdict,
                                    "chart_pattern": getattr(struct, "chart_pattern", "NONE"),
                                    "pricing_zone": getattr(struct, "pricing_zone", "EQUILIBRIUM"),
                                    "playbook": getattr(struct, "playbook", "NONE"),
                                    "signal_score": f"MONITORING ({self._position_health})",
                                }


                                if not struct.is_valid:
                                    logger.warning(
                                        "Structure invalidated for active position",
                                        extra={"symbol": active.symbol},
                                    )
                                    live_price = self._get_live_price(active.symbol)
                                    if live_price > 0:
                                        product = await self.delta_client.get_product(active.symbol)
                                        cv = float(product.get("contract_value", 1.0)) if product else 1.0
                                        pnl = await self.order_manager.close_and_record(
                                            reason="STRUCTURE_INVALIDATED",
                                            close_price=live_price,
                                            contract_value=cv,
                                        )
                                        self.order_manager.clear_closed_orders()
                                        self._last_close_reason = "STRUCTURE_INVALIDATED"
                                        self._position_health = "--"
                                        self._re_entry_cooldown_until = time.time() + 10.0
                                    else:
                                        await self.order_manager.on_structure_invalidated(is_filled=True)
                            except Exception:
                                logger.debug("Structure check failed", exc_info=True)

                else:
                    # ===== MODE A: No Position — Scan & Enter =====

                    # Check re-entry cooldown
                    if time.time() < self._re_entry_cooldown_until:
                        remaining = self._re_entry_cooldown_until - time.time()
                        self._monitored_market["signal_score"] = f"COOLDOWN ({remaining:.0f}s)"
                        continue

                    # Check failsafes
                    failsafe_ok, failures = self.risk_manager.check_failsafes()
                    if not failsafe_ok:
                        logger.info(f"Failsafes blocking entries: {failures}")
                        self._best_signal = None
                        continue

                    setup = await self._scan_assets()
                    self._best_signal = setup

                    if setup:
                        # Store S/R levels from the winning setup
                        self._nearest_support = setup["structure"].nearest_support
                        self._nearest_resistance = setup["structure"].nearest_resistance

                        corr_id = new_correlation_id()
                        logger.info(
                            f"Signal found: {setup['symbol']} {setup['signal'].direction} "
                            f"score={setup['score']:.3f} "
                            f"S={self._nearest_support:.2f} R={self._nearest_resistance:.2f}",
                            extra={"correlation_id": corr_id, "symbol": setup["symbol"]},
                        )

                        # Attempt to place order via OrderManager (handles 9-step validation)
                        delta_live_price = await self.delta_client.fetch_ticker_price(setup["symbol"])
                        if not delta_live_price or delta_live_price <= 0:
                            delta_live_price = self._get_live_price(setup["symbol"], prefer_delta=True)
                        exec_price = delta_live_price if delta_live_price > 0 else setup["signal"].entry_price
                        exec_cfg = getattr(getattr(self.config, "strategy", None), "execution", None)
                        chosen_order_type = getattr(exec_cfg, "order_type", "market") if exec_cfg else "market"
                        try:
                            placed_order = await self.order_manager.validate_and_place_order(
                                symbol=setup["symbol"],
                                side=setup["signal"].direction,
                                entry_price=exec_price,
                                sl_price=None,
                                tp_price=None,
                                atr=setup["atr"],
                                spread_bps=0.0,
                                equity=self.account_manager.equity,
                                order_type=chosen_order_type,
                                structural_target_tp=getattr(setup["signal"], "structural_target_tp", None),
                                setup_type=getattr(setup["signal"], "setup_type", "NONE"),
                                trap_wick_price=getattr(setup["signal"], "trap_wick_price", None),
                                fingerprint=setup.get("fingerprint"),
                                technical_levels=setup.get("technical_levels"),
                                conviction_multiplier=setup.get("conviction_multiplier", 1.0),
                                execution_routing=setup.get("execution_routing", "MAKER_POST_ONLY"),
                            )

                            if placed_order is not None:
                                self._position_health = "STRONG"
                            else:
                                logger.warning(
                                    f"Order placement rejected or not placed by OrderManager for {setup['symbol']}",
                                    extra={"correlation_id": corr_id},
                                )
                                self._re_entry_cooldown_until = time.time() + 15.0
                                self._position_health = "--"
                        except Exception:
                            logger.exception(
                                "Order placement failed",
                                extra={"correlation_id": corr_id},
                            )
                            self._re_entry_cooldown_until = time.time() + 15.0

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Strategy loop error")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Dashboard loop
    # ------------------------------------------------------------------

    async def _dashboard_loop(self) -> None:
        """Periodically render the dashboard."""
        from rich.live import Live

        # Pre-populate dashboard once so the first rendered frame displays full data
        initial_account = self.account_manager.to_dashboard_dict()
        initial_position = self.account_manager.get_position_dict()
        initial_signal = dict(self._monitored_market) if self._monitored_market else {}
        btc_p = self._get_live_price("BTCUSD")
        eth_p = self._get_live_price("ETHUSD")
        if btc_p > 0:
            initial_signal["btc_price"] = btc_p
        if eth_p > 0:
            initial_signal["eth_price"] = eth_p

        if self._best_signal:
            sig = self._best_signal["signal"]
            struct = self._best_signal["structure"]
            initial_signal.update({
                "btc_price": btc_p,
                "eth_price": eth_p,
                "15m_bias": f"[{self._best_signal['symbol']}] {struct.bias_15m}",
                "5m_bos_choch": f"BOS={struct.bos_5m} CHoCH={struct.choch_5m}",
                "liquidity_sweep": str(struct.liquidity_sweep_5m),
                "1m_displacement": str(struct.displacement_1m),
                "retest": str(struct.retest_1m),
                "vwap": getattr(sig, "vwap_position", "--"),
                "rsi": f"{sig.rsi:.1f}",
                "volume": f"{sig.relative_volume:.2f}x",
                "onnx_confidence": f"{self._best_signal.get('ml_confidence', 0):.1%}",
                "llm_status": self.llm_advisor.get_status(),
                "signal_score": f"ACTIONABLE ({self._best_signal['score']:.3f})",
                "next_trigger": self._monitored_market.get("next_trigger", "CONFLUENCE READY TO EXECUTE"),
            })
        initial_risk = self.risk_manager.get_risk_status()
        initial_execution = self.order_manager.get_execution_dict()

        self.dashboard.update(
            account_data=initial_account,
            position_data=initial_position,
            signal_data=initial_signal,
            risk_data=initial_risk,
            execution_data=initial_execution,
            market_watch_data=self._market_watch,
        )

        with Live(self.dashboard.render(), refresh_per_second=1, console=self.dashboard.console) as live:
            while not self._shutdown.is_set():
                try:
                    # Build data dicts
                    account_data = self.account_manager.to_dashboard_dict()
                    position_data = self.account_manager.get_position_dict()

                    # Add S/R levels, health, and fallback order fields to position data
                    if position_data:
                        position_data["nearest_support"] = self._nearest_support
                        position_data["nearest_resistance"] = self._nearest_resistance
                        position_data["health"] = self._position_health
                        position_data["health_reason"] = self._position_health_reason
                        ao = getattr(self.order_manager, "active_order", None)
                        if ao:
                            if not position_data.get("sl") or position_data.get("sl") == 0:
                                position_data["sl"] = ao.sl_price
                            if not position_data.get("tp") or position_data.get("tp") == 0:
                                position_data["tp"] = ao.tp_price
                            if not position_data.get("leverage") or position_data.get("leverage") <= 1:
                                position_data["leverage"] = self.config.risk.leverage.high_leverage_value
                            position_data["sl_anchor"] = getattr(ao, "sl_anchor", None)
                            position_data["tp_target_type"] = getattr(ao, "tp_target_type", None)
                            position_data["target_rr"] = getattr(ao, "target_rr", None)
                            position_data["realized_rr"] = getattr(ao, "realized_rr", None)
                            position_data["cushion_grade"] = getattr(ao, "cushion_grade", None)
                            position_data["execution_routing"] = getattr(ao, "execution_routing", None)

                    # Signal data: use monitored_market (updated by strategy loop during position)
                    signal_data = dict(self._monitored_market) if self._monitored_market else {}

                    # Continuously refresh real-time prices on every tick for all universe assets
                    btc_p = self._get_live_price("BTCUSD")
                    eth_p = self._get_live_price("ETHUSD")
                    if btc_p > 0:
                        signal_data["btc_price"] = btc_p
                    if eth_p > 0:
                        signal_data["eth_price"] = eth_p

                    # Update all assets in market watch
                    assets_dict = self._market_watch.setdefault("assets", {})
                    for sym in self.config.strategy.assets.universe:
                        sym_p = self._get_live_price(sym)
                        sym_dp = self.delta_client.get_latest_price(sym) or sym_p
                        if sym in assets_dict:
                            if sym_p > 0:
                                assets_dict[sym]["price"] = sym_p
                                assets_dict[sym]["delta_price"] = sym_dp
                        elif sym_p > 0:
                            assets_dict[sym] = {
                                "symbol": sym,
                                "price": sym_p,
                                "delta_price": sym_dp,
                            }


                    if self._best_signal and not self.order_manager.active_order:
                        sig = self._best_signal["signal"]
                        struct = self._best_signal["structure"]
                        signal_data.update({
                            "btc_price": btc_p,
                            "eth_price": eth_p,
                            "rankings": self._monitored_market.get("rankings", "--"),
                            "15m_bias": f"[{self._best_signal['symbol']}] {struct.bias_15m}",
                            "5m_bos_choch": f"BOS={struct.bos_5m} CHoCH={struct.choch_5m}",
                            "liquidity_sweep": str(struct.liquidity_sweep_5m),
                            "1m_displacement": str(struct.displacement_1m),
                            "retest": str(struct.retest_1m),
                            "vwap": getattr(sig, "vwap_position", "--"),
                            "rsi": f"{sig.rsi:.1f}",
                            "volume": f"{sig.relative_volume:.2f}x",
                            "onnx_confidence": f"{self._best_signal.get('ml_confidence', 0):.1%}",
                            "llm_status": self.llm_advisor.get_status(),
                            "signal_score": f"ACTIONABLE ({self._best_signal['score']:.3f})",
                            "next_trigger": self._monitored_market.get("next_trigger", "CONFLUENCE READY TO EXECUTE"),
                            "setup_type": getattr(struct, "setup_type", "NONE"),
                            "pattern": getattr(sig, "pattern", "NONE"),
                        })

                    risk_data = self.risk_manager.get_risk_status()

                    execution_data = self.order_manager.get_execution_dict()

                    self.dashboard.update(
                        account_data=account_data,
                        position_data=position_data,
                        signal_data=signal_data,
                        risk_data=risk_data,
                        execution_data=execution_data,
                        market_watch_data=self._market_watch,
                    )
                    live.update(self.dashboard.render())
                except Exception as e:
                    logger.debug(f"Dashboard render loop tick error: {e}", exc_info=True)

                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main async entry point."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows

        logger.info(f"Starting scalping bot in {self.mode.upper()} mode")

        # Kill switch check
        if self.risk_manager.is_kill_switch_active:
            logger.warning("Kill switch is active — no new entries will be placed")

        # Daily reset check
        self.risk_manager.reset_daily()

        # Pre-seed deep historical market data (3000+ candles from 5s to HTF)
        logger.info("Pre-seeding deep historical market data (3000+ candles) from Delta Exchange...")
        try:
            bootstrapped = await self.delta_ws.bootstrap_historical_candles(limit=3000)
            await self.delta_ws.bootstrap_htf_candles(limit=168)
            logger.info(f"Market data bootstrapped: {bootstrapped} candles loaded across multiple timeframes.")
        except Exception as e:
            logger.warning(f"Candle bootstrapping issue: {e}")

        # Fetch official Delta product specifications for BTC and ETH only
        logger.info("Fetching Delta Exchange product specifications for BTC and ETH...")
        try:
            delta_prods = await self.delta_client.fetch_target_products(["BTCUSD", "ETHUSD"])
            logger.info(f"Delta products loaded: {list(delta_prods.keys())}")
        except Exception as e:
            logger.warning(f"Delta product bootstrap note: {e}")

        # Initial account balance and positions fetch
        if self.is_live:
            try:
                balances = await self.delta_client.get_wallet_balances()
                positions = await self.delta_client.get_positions()
                self.account_manager.update_from_exchange(balances, positions)
                self.risk_manager.delta_connected = True
                self.risk_manager.delta_error_reason = ""
                logger.info(f"Delta Exchange connected: Equity=${self.account_manager.equity:,.2f}")
            except Exception as e:
                self.risk_manager.delta_connected = False
                err_str = str(e)
                if "ip_not_whitelisted" in err_str.lower():
                    self.risk_manager.delta_error_reason = "IP_NOT_WHITELISTED"
                    logger.error(
                        "DELTA LIVE API ERROR: Your IP is not whitelisted for this Delta API key. "
                        "Please add your IP in Delta Exchange API settings or create a non-IP-restricted key."
                    )
                else:
                    self.risk_manager.delta_error_reason = "DELTA_DISCONNECTED"
                    logger.warning(f"Delta Exchange initial account check failed: {e}")

        self.account_manager.reset_daily(self.account_manager.equity)

        # Pre-seed Delta tickers
        for s in ("BTCUSD", "ETHUSD"):
            try:
                await self.delta_client.fetch_ticker_price(s)
            except Exception:
                pass

        # Initial scan so monitored market overview is populated before first dashboard frame
        try:
            self._best_signal = await self._scan_assets()
        except Exception as e:
            logger.debug(f"Initial scan error: {e}")

        tasks = []

        if self.mode in ("paper", "live"):
            tasks.append(asyncio.create_task(self.delta_client.run_public_ticker(list(self.config.strategy.assets.universe))))
            tasks.append(asyncio.create_task(self._run_delta_ws()))
            tasks.append(asyncio.create_task(self._strategy_loop()))
            tasks.append(asyncio.create_task(self._position_monitor_loop()))
            tasks.append(asyncio.create_task(self._dashboard_loop()))

        elif self.mode == "scan":
            # One-shot scan: candles already bootstrapped!
            setup = await self._scan_assets()
            if setup:
                print(f"\n🎯 Best setup: {setup['symbol']} {setup['signal'].direction}")
                print(f"   Score: {setup['score']:.3f}")
                print(f"   ML Confidence: {setup.get('ml_confidence', 0):.1%}")
            else:
                top_sym = self._monitored_market.get("15m_bias", "BTCUSD")
                rankings = self._monitored_market.get("rankings", "--")
                print(f"\n📊 Monitored Market: {top_sym}")
                print(f"   Rankings: {rankings}")
                print(f"   5M Structure: {self._monitored_market.get('5m_bos_choch')}")
                print(f"   RSI: {self._monitored_market.get('rsi')} | VWAP: {self._monitored_market.get('vwap')}")
                print(f"⏳ No confluence trigger at this second. Scanner is operational and monitoring all {len(self.config.strategy.assets.universe)} assets.")
            self._shutdown.set()

        elif self.mode == "status":
            # Show current account status
            try:
                if self.is_live:
                    balances = await self.delta_client.get_wallet_balances()
                    print(f"\n💰 Delta Live Equity: ${balances.get('equity', 0):,.2f}")
                    print(f"   Available Margin: ${balances.get('available_balance', 0):,.2f}")
                    print(f"   Position Margin:  ${balances.get('position_margin', 0):,.2f}")
                    positions = await self.delta_client.get_positions()
                    if positions:
                        for pos in positions:
                            print(f"📊 Position: {pos}")
                    else:
                        print("📊 Open Positions: None (0 active)")
                else:
                    print(f"\n📋 Paper Account Balance: ${self.account_manager.equity:,.2f}")
            except Exception as e:
                print(f"❌ Error: {e}")
            self._shutdown.set()

        elif self.mode == "backtest":
            print("\n📈 Backtest mode is a placeholder for future implementation.")
            print("   Historical data replay will be added in a future release.")
            self._shutdown.set()

        if tasks:
            # Wait for shutdown
            await self._shutdown.wait()
            logger.info("Shutting down...")

            # Cancel all tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup clients
        try:
            await self.delta_ws.stop()
        except Exception:
            pass
        try:
            await self.delta_client.close()
        except Exception:
            pass
        try:
            await self.jev_client.close()
        except Exception:
            pass

        logger.info("Bot stopped")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config = load_config(config_dir=getattr(args, "config_dir", None))

    # Override log level if specified via CLI
    log_level = args.log_level or config.env.log_level
    setup_logging(
        level=log_level,
        log_dir=config.env.log_dir,
        console=(args.mode not in ("paper", "live")),  # Dashboard handles console in paper/live
    )

    logger.info(f"Config loaded: {config.env}")

    # CLI overrides for target trading universe
    if getattr(args, "symbols", None):
        from src.execution.delta import normalize_delta_symbol
        raw_syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        normalized_syms = [normalize_delta_symbol(s) for s in raw_syms]
        if normalized_syms:
            config.strategy.assets.universe = normalized_syms
            logger.info(f"Target universe overridden via --symbols: {normalized_syms}")
    elif getattr(args, "universe", None):
        preset_name = args.universe.strip().lower()
        presets = getattr(config.strategy.assets, "presets", {})
        if presets and preset_name in presets:
            config.strategy.assets.universe = list(presets[preset_name])
            logger.info(f"Target universe preset '{preset_name}' applied: {config.strategy.assets.universe}")
        else:
            logger.warning(f"Universe preset '{preset_name}' not found. Available: {list(presets.keys()) if presets else 'None'}")

    if args.kill_switch:
        logger.warning("Kill switch activated via CLI")

    bot = ScalpingBot(
        config=config,
        mode=args.mode,
        paper_balance=args.paper_balance,
    )

    if args.kill_switch:
        bot.risk_manager.activate_kill_switch()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
