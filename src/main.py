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
from src.execution.delta import DeltaExchangeClient
from src.execution.order_manager import OrderManager, OrderState
from src.strategy.structure import MarketStructure
from src.strategy.signals import SignalGenerator
from src.strategy.regime import RegimeFilter
from src.ml.onnx_model import ONNXScalperModel
from src.llm.advisor import LLMAdvisor
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
        target_symbols = ["BTCUSD", "ETHUSD"]
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

        # ---- LLM layer ----
        self.llm_advisor = LLMAdvisor(
            api_key=config.env.litellm_api_key,
            model=config.env.litellm_model,
        )

        # ---- Risk layer ----
        self.risk_manager = RiskManager(config.risk)

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
        )

        # ---- UI layer ----
        self.dashboard = Dashboard(mode=mode)

        # ---- State ----
        self._last_scan_time: float = 0.0
        self._best_signal: dict[str, Any] | None = None
        self._position_health: str = "--"
        self._position_health_reason: str = ""
        self._nearest_support: float = 0.0
        self._nearest_resistance: float = 0.0
        self._last_close_reason: str = ""
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

                if position_age >= max_holding_sec:
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

                # --- 1b. Dynamic Stepped Trailing Stop Loss on Margin P&L ---
                if product_id:
                    product = await self.delta_client.get_product(symbol)
                    cv = float(product.get("contract_value", 1.0)) if product else 1.0
                    tick_size = float(product.get("tick_size", 0.0)) if product else None
                    pos_entry = self.account_manager.positions.get(symbol) if self.account_manager else None
                    margin = pos_entry.margin if (pos_entry and pos_entry.margin > 0) else (active.entry_price * active.size * cv / max(1, self.config.risk.leverage.high_leverage_value))

                    new_sl, updated, msg = self.risk_manager.calculate_trailing_stop_loss(
                        entry_price=active.entry_price,
                        side=active.side,
                        margin=margin,
                        current_price=live_price,
                        contract_value=cv,
                        size=int(active.size),
                        current_sl=active.sl_price or 0.0,
                        tick_size=tick_size,
                    )
                    if updated:
                        self.order_manager.update_active_sl(new_sl)
                        await self.delta_client.update_bracket_stop_loss(product_id, new_sl, tick_size, order_id=active.order_id)
                        logger.info(f"[TRAILING SL] {symbol}: {msg}")

                # --- 2. Check paper SL/TP triggers ---
                if not self.delta_client.live_trading and product_id:
                    sl_tp_result = self.delta_client.check_paper_sl_tp(product_id, live_price)
                    if sl_tp_result:
                        reason = sl_tp_result  # "SL_HIT" or "TP_HIT"
                        logger.info(f"Paper {reason} for {symbol} at {live_price:.2f}")

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

                        # Set re-entry cooldown (5s for TP, 30s for SL)
                        cooldown = 5.0 if reason == "TP_HIT" else float(self.config.risk.daily_limits.cooldown_seconds)
                        if pnl >= 0:
                            cooldown = 5.0  # Quick re-entry on profit
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

            try:
                structure = self.market_structure.analyze(
                    arr_15m, arr_5m, arr_1m, current_atr
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

            # ML confirmation
            ml_confirmed = True
            ml_confidence = 0.0
            if self.config.strategy.ml.enabled and self.onnx_model.is_available:
                vwap_values = compute_vwap(arr_1m["high"], arr_1m["low"], arr_1m["close"], arr_1m["volume"])
                cur_vwap = float(vwap_values[-1]) if len(vwap_values) > 0 else float(arr_1m["close"][-1])
                vwap_dist = (float(arr_1m["close"][-1]) - cur_vwap) / cur_vwap if cur_vwap > 0 else 0.0

                features = self.onnx_model.prepare_features(
                    ema_cross_signal=sig.ema_cross,
                    rsi=sig.rsi,
                    atr_normalized=current_atr / float(arr_1m["close"][-1]) if arr_1m["close"][-1] > 0 else 0,
                    vwap_distance=vwap_dist,
                    volume_ratio=sig.relative_volume,
                    adx=current_adx,
                    structure_score=sig.strength,
                )
                eval_dir = sig.direction if sig.direction != "NONE" else ("LONG" if structure.bias_15m == "BULLISH" else "SHORT")
                ml_confirmed, ml_confidence = self.onnx_model.confirm_signal(
                    eval_dir, features, self.config.strategy.ml.min_confidence
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

            is_actionable = structure.is_valid and sig.direction in ("LONG", "SHORT") and (ml_confirmed or score >= 0.55)

            evaluations.append({
                "symbol": symbol,
                "signal": sig,
                "structure": structure,
                "regime": regime,
                "ml_confidence": ml_confidence,
                "ml_confirmed": ml_confirmed,
                "score": score,
                "atr": current_atr,
                "is_actionable": is_actionable,
            })

        if not evaluations:
            return None

        # Sort rankings: Actionable triggers first, then by highest score
        evaluations.sort(key=lambda x: (1 if x["is_actionable"] else 0, x["score"]), reverse=True)
        self._asset_rankings = evaluations

        # Formatted ranking summary string for dashboard: e.g. "#1 BTCUSD: 0.78 (LONG) | #2 ETHUSD: 0.62 (NONE)"
        rankings_str = " | ".join(
            [f"#{i+1} {e['symbol']} ({e['score']:.2f}, {e['signal'].direction})" for i, e in enumerate(evaluations)]
        )

        top = evaluations[0]
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
                "actionable": e["is_actionable"],
                "setup_type": getattr(e["structure"], "setup_type", "NONE"),
                "pattern": getattr(e["signal"], "pattern", "NONE"),
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
            "llm_status": self.llm_advisor.get_status(),
            "signal_score": f"ACTIONABLE ({top['score']:.3f})" if top["is_actionable"] else f"RANK #{1} {top['symbol']} ({top['score']:.2f})",
            "next_trigger": next_trigger,
            "setup_type": getattr(top["structure"], "setup_type", "NONE"),
            "pattern": getattr(top["signal"], "pattern", "NONE"),
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

                            try:
                                struct = self.market_structure.analyze(
                                    to_arr(candles_15m), to_arr(candles_5m), a1m, cur_atr
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
                                    "llm_status": self.llm_advisor.get_status(),
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
                            await self.order_manager.validate_and_place_order(
                                symbol=setup["symbol"],
                                side=setup["signal"].direction,
                                entry_price=exec_price,
                                sl_price=None,
                                tp_price=None,
                                atr=setup["atr"],
                                spread_bps=0.0,
                                equity=self.account_manager.equity,
                                order_type=chosen_order_type,
                            )

                            self._position_health = "STRONG"
                        except Exception:
                            logger.exception(
                                "Order placement failed",
                                extra={"correlation_id": corr_id},
                            )

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

                    # Signal data: use monitored_market (updated by strategy loop during position)
                    signal_data = dict(self._monitored_market) if self._monitored_market else {}

                    # Continuously refresh real-time prices on every tick
                    btc_p = self._get_live_price("BTCUSD")
                    eth_p = self._get_live_price("ETHUSD")
                    btc_delta = self.delta_client.get_latest_price("BTCUSD") or btc_p
                    eth_delta = self.delta_client.get_latest_price("ETHUSD") or eth_p
                    if btc_p > 0:
                        signal_data["btc_price"] = btc_p
                        if "BTCUSD" in self._market_watch.get("assets", {}):
                            self._market_watch["assets"]["BTCUSD"]["price"] = btc_p
                            self._market_watch["assets"]["BTCUSD"]["delta_price"] = btc_delta
                    if eth_p > 0:
                        signal_data["eth_price"] = eth_p
                        if "ETHUSD" in self._market_watch.get("assets", {}):
                            self._market_watch["assets"]["ETHUSD"]["price"] = eth_p
                            self._market_watch["assets"]["ETHUSD"]["delta_price"] = eth_delta


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

        # Pre-seed historical market data for immediate strategy execution
        logger.info("Pre-seeding historical market data from Delta Exchange...")
        try:
            bootstrapped = await self.delta_ws.bootstrap_historical_candles(limit=100)
            logger.info(f"Market data bootstrapped: {bootstrapped} candles loaded.")
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
            tasks.append(asyncio.create_task(self.delta_client.run_public_ticker(["BTCUSD", "ETHUSD"])))
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
