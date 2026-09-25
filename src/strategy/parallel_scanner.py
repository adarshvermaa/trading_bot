"""Parallel Multi-Worker Scanner & Asset Intelligence Agent Architecture.

Executes concurrent, non-blocking technical analysis, market structure evaluation,
and confluence checks across the crypto universe using asynchronous sub-workers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.delta_ws import Candle, CandleStore, DeltaWSClient
from src.strategy.signals import (
    Signal,
    SignalGenerator,
    compute_adx,
    compute_atr,
    compute_cvd,
    compute_order_flow_imbalance,
    compute_squeeze_momentum,
    compute_vwap_bands,
)
from src.strategy.structure import MarketStructure, StructureAnalysis
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _candles_to_arrays(candles: List[Candle]) -> Dict[str, np.ndarray]:
    """Convert candle list to vectorized numpy arrays."""
    if not candles:
        return {
            "open": np.array([]),
            "high": np.array([]),
            "low": np.array([]),
            "close": np.array([]),
            "volume": np.array([]),
        }
    return {
        "open": np.array([c.open for c in candles]),
        "high": np.array([c.high for c in candles]),
        "low": np.array([c.low for c in candles]),
        "close": np.array([c.close for c in candles]),
        "volume": np.array([c.volume for c in candles]),
    }


@dataclass
class AssetEvaluation:
    """Standardized output from an AssetWorker."""
    symbol: str
    signal: Signal
    structure: StructureAnalysis
    score: float
    current_price: float
    current_atr: float
    spread_bps: float
    cvd_divergence: str
    squeeze_state: str
    ofi_score: float
    htf_bias: str
    timeframes_present: List[str]
    latency_ms: float
    raw_arrays: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


class AssetWorker:
    """Dedicated asynchronous sub-agent that analyzes a single asset in parallel."""

    def __init__(
        self,
        symbol: str,
        candle_store: CandleStore,
        market_structure: MarketStructure,
        signal_generator: SignalGenerator,
        delta_client: Optional[Any] = None,
        delta_ws: Optional[DeltaWSClient] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.symbol = symbol.upper()
        self.candle_store = candle_store
        self.market_structure = market_structure
        self.signal_generator = signal_generator
        self.delta_client = delta_client
        self.delta_ws = delta_ws
        self.weights = weights or {
            "structure": 0.35,
            "regime": 0.20,
            "indicators": 0.25,
            "ml_confidence": 0.20,
        }

    async def evaluate(self) -> Optional[AssetEvaluation]:
        """Perform comprehensive multi-timeframe analysis on the asset."""
        t_start = time.perf_counter()
        sym = self.symbol

        # Query all available timeframes (micro, intermediate, macro)
        all_tfs = ["5s", "15s", "30s", "1m", "3m", "5m", "15m", "30m", "45m", "1h", "4h", "1d"]
        c_map: Dict[str, List[Candle]] = {}
        for tf in all_tfs:
            candles = self.candle_store.get_candles(sym, tf)
            if candles:
                c_map[tf] = candles

        # Require at least primary structure and execution candles (1m, 5m, 15m)
        c_1m = c_map.get("1m", [])
        c_5m = c_map.get("5m", [])
        c_15m = c_map.get("15m", [])

        if len(c_1m) < 20 or len(c_5m) < 20 or len(c_15m) < 20:
            return None

        # Vectorize arrays
        arr_1m = _candles_to_arrays(c_1m)
        arr_5m = _candles_to_arrays(c_5m)
        arr_15m = _candles_to_arrays(c_15m)

        # Micro triggers if available
        c_30s = c_map.get("30s", [])
        c_15s = c_map.get("15s", [])
        c_5s = c_map.get("5s", [])

        # Current Price & ATR
        current_price = float(arr_1m["close"][-1]) if len(arr_1m["close"]) > 0 else 0.0
        if self.delta_ws:
            live_price = self.delta_ws.get_latest_price(sym)
            if live_price and live_price > 0:
                current_price = live_price

        atr_arr = compute_atr(arr_1m["high"], arr_1m["low"], arr_1m["close"], 14)
        current_atr = float(atr_arr[-1]) if len(atr_arr) > 0 else (current_price * 0.005)

        # Spread check
        spread_bps = 5.0
        quotes = self.delta_ws.get_latest_quotes(sym) if self.delta_ws else None
        if quotes and "best_bid" in quotes and "best_ask" in quotes:
            try:
                bid = float(quotes["best_bid"])
                ask = float(quotes["best_ask"])
                if bid > 0 and ask > bid:
                    spread_bps = ((ask - bid) / bid) * 10_000.0
            except (ValueError, TypeError):
                pass

        # Order Flow Imbalance (OFI) from live L2 book if available
        ofi_score = 0.0
        if self.delta_client and hasattr(self.delta_client, "get_orderbook"):
            try:
                ob = await self.delta_client.get_orderbook(sym, depth=10)
                if ob:
                    bids = ob.get("buy", ob.get("bids", []))
                    asks = ob.get("sell", ob.get("asks", []))
                    ofi_score = compute_order_flow_imbalance(bids, asks)
            except Exception:
                pass

        # Higher Timeframe Levels (PDH, PDL, POC, VAH, VAL)
        htf_levels = self.delta_ws.get_htf_levels(sym) if self.delta_ws else {}

        # 1. Market Structure Analysis
        structure = self.market_structure.analyze(
            arr_15m, arr_5m, arr_1m, current_atr, htf_levels=htf_levels
        )

        # 2. Confluence Signal Generation
        signal = self.signal_generator.generate(
            arr_15m, arr_5m, arr_1m, structure
        )
        signal.ofi_score = ofi_score
        signal.timeframes_analyzed = list(c_map.keys())

        # Micro-timeframe trigger confirmation (30s/15s/5s)
        if c_30s and len(c_30s) >= 10:
            arr_30s = _candles_to_arrays(c_30s)
            micro_atr = compute_atr(arr_30s["high"], arr_30s["low"], arr_30s["close"], 14)
            if len(micro_atr) > 0 and signal.direction == "LONG":
                # Micro bounce confirmation
                if arr_30s["close"][-1] > arr_30s["open"][-1]:
                    signal.strength = min(0.99, signal.strength + 0.03)
            elif len(micro_atr) > 0 and signal.direction == "SHORT":
                # Micro drop confirmation
                if arr_30s["close"][-1] < arr_30s["open"][-1]:
                    signal.strength = min(0.99, signal.strength + 0.03)

        # OFI Confluence adjustment
        if signal.direction == "LONG" and ofi_score > 0.20:
            signal.strength = min(0.99, signal.strength + 0.03)
        elif signal.direction == "SHORT" and ofi_score < -0.20:
            signal.strength = min(0.99, signal.strength + 0.03)

        # Calculate composite score
        struct_score = 1.0 if structure.is_valid else 0.5
        if getattr(structure, "setup_type", "NONE") != "NONE":
            struct_score = 0.95
        ind_score = signal.strength
        regime_score = 0.80

        w_struct = self.weights.get("structure", 0.35)
        w_ind = self.weights.get("indicators", 0.25)
        w_regime = self.weights.get("regime", 0.20)
        total_w = w_struct + w_ind + w_regime
        composite_score = (
            (w_struct * struct_score) + (w_ind * ind_score) + (w_regime * regime_score)
        ) / (total_w if total_w > 0 else 1.0)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return AssetEvaluation(
            symbol=sym,
            signal=signal,
            structure=structure,
            score=composite_score,
            current_price=current_price,
            current_atr=current_atr,
            spread_bps=spread_bps,
            cvd_divergence=signal.cvd_divergence,
            squeeze_state=signal.squeeze_state,
            ofi_score=ofi_score,
            htf_bias=structure.bias_15m,
            timeframes_present=list(c_map.keys()),
            latency_ms=latency_ms,
            raw_arrays={"1m": arr_1m, "5m": arr_5m, "15m": arr_15m},
        )


class ParallelScanner:
    """Orchestrates concurrent analysis across the universe of assets."""

    def __init__(
        self,
        universe: List[str],
        candle_store: CandleStore,
        market_structure: MarketStructure,
        signal_generator: SignalGenerator,
        delta_client: Optional[Any] = None,
        delta_ws: Optional[DeltaWSClient] = None,
        weights: Optional[Dict[str, float]] = None,
        max_concurrent_workers: int = 10,
    ):
        self.universe = [s.upper() for s in universe]
        self.candle_store = candle_store
        self.market_structure = market_structure
        self.signal_generator = signal_generator
        self.delta_client = delta_client
        self.delta_ws = delta_ws
        self.weights = weights
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)

        # Worker map: symbol -> AssetWorker
        self.workers: Dict[str, AssetWorker] = {
            sym: AssetWorker(
                symbol=sym,
                candle_store=candle_store,
                market_structure=market_structure,
                signal_generator=signal_generator,
                delta_client=delta_client,
                delta_ws=delta_ws,
                weights=weights,
            )
            for sym in self.universe
        }

    async def _run_worker(self, worker: AssetWorker) -> Optional[AssetEvaluation]:
        async with self.semaphore:
            try:
                return await worker.evaluate()
            except Exception as e:
                logger.debug(f"Worker for {worker.symbol} encountered an error: {e}")
                return None

    async def scan(self, target_symbols: Optional[List[str]] = None) -> List[AssetEvaluation]:
        """
        Execute concurrent scan across all requested universe assets.
        Returns a sorted list of actionable AssetEvaluations (highest score first).
        """
        t0 = time.perf_counter()
        targets = [s.upper() for s in target_symbols] if target_symbols else self.universe

        tasks = []
        for sym in targets:
            worker = self.workers.get(sym)
            if not worker:
                worker = AssetWorker(
                    symbol=sym,
                    candle_store=self.candle_store,
                    market_structure=self.market_structure,
                    signal_generator=self.signal_generator,
                    delta_client=self.delta_client,
                    delta_ws=self.delta_ws,
                    weights=self.weights,
                )
                self.workers[sym] = worker
            tasks.append(self._run_worker(worker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_evaluations: List[AssetEvaluation] = []
        for res in results:
            if isinstance(res, AssetEvaluation) and res is not None:
                valid_evaluations.append(res)
            elif isinstance(res, Exception):
                logger.debug(f"Parallel scan task failed: {res}")

        # Sort by composite score descending
        valid_evaluations.sort(key=lambda x: x.score, reverse=True)
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        if valid_evaluations:
            logger.debug(
                f"[PARALLEL SCAN] Evaluated {len(valid_evaluations)}/{len(targets)} assets in {total_time_ms:.1f}ms. "
                f"Top: {valid_evaluations[0].symbol} ({valid_evaluations[0].signal.direction} score={valid_evaluations[0].score:.2f})"
            )

        return valid_evaluations
