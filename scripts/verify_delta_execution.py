#!/usr/bin/env python3
"""Delta Exchange API & Futures Execution Verification Tool.

Verifies:
1. Futures product universe & live tickers (BTCUSD, ETHUSD, XAUTUSD, etc.)
2. Tick-size precision rounding for Entry, Stop Loss, and Take Profit
3. Futures wallet balances & available margin (private API)
4. Open futures positions & open orders (private API)
5. Optional live bracket order placement & cancellation verification (--test-order)

Usage:
    # Public endpoints + SL/TP rounding verification (no keys needed):
    python scripts/verify_delta_execution.py --dry-run

    # Full verification using .env credentials:
    python scripts/verify_delta_execution.py

    # Full verification with explicit credentials:
    python scripts/verify_delta_execution.py --api-key YOUR_KEY --api-secret YOUR_SECRET

    # Test safe out-of-the-money bracket order placement & immediate cancellation:
    python scripts/verify_delta_execution.py --test-order
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

from src.risk.risk_manager import round_to_tick, format_price, RiskManager
from src.config import load_config, RiskConfig

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False


TOP_10_UNIVERSE = [
    "BTCUSD", "ETHUSD", "XAUTUSD", "PAXGUSD", "SOLUSD",
    "XRPUSD", "BNBUSD", "DOGEUSD", "TRXUSD", "HYPEUSD"
]


def mask_secret(secret: str) -> str:
    if not secret:
        return "[red]EMPTY / NOT SET[/red]"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]} (len={len(secret)})"


class DeltaVerifier:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def init(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _generate_signature(self, method: str, timestamp: str, path: str, query_string: str = "", body: str = "") -> str:
        payload = method + timestamp + path + query_string + body
        return hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def request(
        self, 
        method: str, 
        path: str, 
        params: Optional[Dict[str, Any]] = None, 
        body: Optional[Dict[str, Any]] = None,
        auth_required: bool = False
    ) -> Tuple[bool, Any, int]:
        await self.init()
        timestamp = str(int(time.time()))
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())
            
        body_str = json.dumps(body) if body else ""
        url = self.base_url + path + query_string
        
        headers = {
            "User-Agent": "delta-futures-verifier/1.0",
            "Content-Type": "application/json"
        }
        
        if auth_required:
            if not self.api_key or not self.api_secret:
                return False, {"error": "API Key and Secret are required for this endpoint"}, 401
            signature = self._generate_signature(method, timestamp, path, query_string, body_str)
            headers["api-key"] = self.api_key
            headers["signature"] = signature
            headers["timestamp"] = timestamp
            
        try:
            async with self.session.request(method, url, headers=headers, data=body_str if body else None, timeout=10) as resp:
                status = resp.status
                try:
                    data = await resp.json()
                except Exception:
                    text = await resp.text()
                    return False, {"raw_response": text}, status
                    
                success = data.get("success", False) if isinstance(data, dict) else False
                return success, data.get("result", data), status
        except Exception as e:
            return False, {"error": str(e)}, 0


async def verify_public_futures(verifier: DeltaVerifier) -> Dict[str, Dict[str, Any]]:
    """Verify that all universe products are active futures and fetch their metadata."""
    print("\n" + "=" * 70)
    print("STEP 1: Verifying Delta Exchange Futures Products & Tickers")
    print("=" * 70)

    success, products, status = await verifier.request(
        "GET", 
        "/v2/products", 
        params={"contract_types": "perpetual_futures,futures", "states": "live"}
    )
    
    if not success or not isinstance(products, list):
        print(f"❌ Failed to fetch products from Delta API: HTTP {status}, response: {products}")
        return {}

    products_by_sym = {p["symbol"]: p for p in products if p.get("contract_type") in ("perpetual_futures", "futures")}
    
    table_data = []
    found_assets = {}
    
    for sym in TOP_10_UNIVERSE:
        p = products_by_sym.get(sym)
        if not p:
            table_data.append((sym, "NOT FOUND", "-", "-", "-", "-", "-"))
            continue
            
        pid = p["id"]
        c_type = p.get("contract_type", "futures")
        c_val = float(p.get("contract_value", 1.0))
        tick = float(p.get("tick_size", 0.1))
        init_margin = float(p.get("initial_margin", 0.01))
        if init_margin > 0:
            max_lev = round(100.0 / init_margin if init_margin >= 0.05 else 1.0 / init_margin, 1)
        else:
            max_lev = 1.0
        
        # Fetch ticker
        t_ok, ticker, _ = await verifier.request("GET", f"/v2/tickers/{sym}")
        mark_price = float(ticker.get("mark_price", 0.0)) if t_ok and isinstance(ticker, dict) else 0.0
        
        found_assets[sym] = {
            "product": p,
            "product_id": pid,
            "contract_type": c_type,
            "contract_value": c_val,
            "tick_size": tick,
            "initial_margin": init_margin,
            "max_leverage": max_lev,
            "mark_price": mark_price
        }
        
        table_data.append((
            sym, str(pid), c_type, f"{c_val}", f"{tick}", f"{max_lev}x", f"${mark_price:,.4f}" if mark_price else "N/A"
        ))

    if HAS_RICH:
        table = Table(title="Delta Exchange India — Verified Futures Universe", border_style="cyan")
        table.add_column("Symbol", style="bold green")
        table.add_column("Product ID", style="magenta")
        table.add_column("Contract Type", style="yellow")
        table.add_column("Contract Val", style="white")
        table.add_column("Tick Size", style="cyan")
        table.add_column("Max Lev", style="bold red")
        table.add_column("Mark Price", style="bold white")
        for row in table_data:
            table.add_row(*row)
        console.print(table)
    else:
        print(f"{'Symbol':<10} {'ID':<8} {'Type':<18} {'Contract Val':<12} {'Tick':<10} {'Max Lev':<8} {'Mark Price'}")
        print("-" * 75)
        for row in table_data:
            print(f"{row[0]:<10} {row[1]:<8} {row[2]:<18} {row[3]:<12} {row[4]:<10} {row[5]:<8} {row[6]}")

    missing = [s for s in TOP_10_UNIVERSE if s not in found_assets]
    if missing:
        print(f"\n⚠️  WARNING: Missing assets from universe: {missing}")
    else:
        print(f"\n✅ All 10 assets verified as active perpetual futures contracts.")
        
    return found_assets


def verify_rounding_and_sl_tp(found_assets: Dict[str, Dict[str, Any]]):
    """Verify precision tick rounding for Entry, Stop Loss, and Take Profit."""
    print("\n" + "=" * 70)
    print("STEP 2: Verifying Precision & Rounded SL/TP Calculations")
    print("=" * 70)
    
    rm = RiskManager(RiskConfig())
    rounding_data = []
    all_valid = True
    
    for sym, data in found_assets.items():
        tick_size = data["tick_size"]
        mark_price = data["mark_price"]
        if not mark_price or mark_price <= 0:
            mark_price = 100.0  # Fallback dummy price
            
        contract_val = data["contract_value"]
        leverage = min(50, int(data["max_leverage"]))
        size = max(1, int(1000.0 * leverage / (mark_price * contract_val)))
        
        # Round entry price
        entry_price = round_to_tick(mark_price, tick_size, 'NEAREST')
        
        # Calculate SL & TP with tick_size rounding
        sl_price, tp_price = rm.calculate_sl_tp(
            side="LONG",
            entry_price=entry_price,
            margin=1000.0,
            leverage=leverage,
            contract_value=contract_val,
            size=size,
            atr=entry_price * 0.01,
            tick_size=tick_size
        )
        
        # Validate exact tick division
        sl_ticks = round(sl_price / tick_size, 6)
        tp_ticks = round(tp_price / tick_size, 6)
        is_sl_tick_valid = abs(sl_ticks - round(sl_ticks)) < 1e-5
        is_tp_tick_valid = abs(tp_ticks - round(tp_ticks)) < 1e-5
        
        if not (is_sl_tick_valid and is_tp_tick_valid):
            all_valid = False
            
        formatted_entry = format_price(entry_price, tick_size)
        formatted_sl = format_price(sl_price, tick_size)
        formatted_tp = format_price(tp_price, tick_size)
        
        rounding_data.append((
            sym,
            f"{tick_size}",
            formatted_entry,
            formatted_sl,
            formatted_tp,
            f"-{((entry_price - sl_price) / entry_price) * 100:.2f}%",
            f"+{((tp_price - entry_price) / entry_price) * 100:.2f}%",
            "✅ VALID" if (is_sl_tick_valid and is_tp_tick_valid) else "❌ INVALID"
        ))

    if HAS_RICH:
        table = Table(title="Stop Loss / Take Profit Rounded to Valid Exchange Ticks", border_style="green")
        table.add_column("Symbol", style="bold")
        table.add_column("Tick Size", style="cyan")
        table.add_column("Entry Price", style="white")
        table.add_column("Stop Loss (SL)", style="bold red")
        table.add_column("Take Profit (TP)", style="bold green")
        table.add_column("SL Move", style="red")
        table.add_column("TP Move", style="green")
        table.add_column("Tick Aligned", style="bold yellow")
        for row in rounding_data:
            table.add_row(*row)
        console.print(table)
    else:
        print(f"{'Symbol':<10} {'Tick':<10} {'Entry':<12} {'SL Price':<12} {'TP Price':<12} {'SL Move':<8} {'TP Move':<8} {'Tick Check'}")
        print("-" * 80)
        for row in rounding_data:
            print(f"{row[0]:<10} {row[1]:<10} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<8} {row[6]:<8} {row[7]}")

    if all_valid:
        print("\n✅ All Stop Loss & Take Profit levels are strictly aligned to exchange tick size.")
    else:
        print("\n❌ Warning: Tick alignment error detected in calculations!")


async def verify_private_wallet_and_trading(verifier: DeltaVerifier) -> bool:
    """Verify private API endpoints: Wallet balances, open positions, open orders."""
    print("\n" + "=" * 70)
    print("STEP 3: Verifying Delta Exchange Futures Wallet & Trading Endpoints")
    print("=" * 70)
    
    if not verifier.api_key or not verifier.api_secret:
        print("⚠️  SKIPPED: DELTA_API_KEY and DELTA_API_SECRET are not provided.")
        print("   To test private wallet and positions, add your keys to .env or pass via --api-key and --api-secret.")
        return False
        
    print("Authenticating with HMAC-SHA256 signature...")
    
    # 1. Wallet Balances
    b_ok, balances, b_status = await verifier.request("GET", "/v2/wallet/balances", auth_required=True)
    if not b_ok:
        print(f"❌ Failed to fetch wallet balances: HTTP {b_status}, error: {balances}")
        return False
        
    print("✅ Successfully authenticated! Live Wallet Balances retrieved:")
    
    if isinstance(balances, list):
        if HAS_RICH:
            bal_table = Table(title="Delta Exchange India — Futures Wallet Balances", border_style="blue")
            bal_table.add_column("Asset", style="bold white")
            bal_table.add_column("Equity", style="bold green")
            bal_table.add_column("Available Balance", style="bold cyan")
            bal_table.add_column("Order Margin", style="yellow")
            bal_table.add_column("Position Margin", style="magenta")
            
            for b in balances:
                asset = b.get("asset_symbol", "USD")
                equity = float(b.get("equity", b.get("balance", 0.0)))
                avail = float(b.get("available_balance", 0.0))
                order_margin = float(b.get("order_margin", 0.0))
                pos_margin = float(b.get("position_margin", 0.0))
                bal_table.add_row(
                    asset, f"{equity:,.2f}", f"{avail:,.2f}", f"{order_margin:,.2f}", f"{pos_margin:,.2f}"
                )
            console.print(bal_table)
        else:
            for b in balances:
                print(f"   Asset: {b.get('asset_symbol')} | Equity: {b.get('equity')} | Available: {b.get('available_balance')} | Margin: {b.get('position_margin')}")
    else:
        print(f"   Balances: {balances}")

    # 2. Open Futures Positions
    p_ok, positions, p_status = await verifier.request(
        "GET", "/v2/positions/margined", params={"contract_types": "perpetual_futures,futures"}, auth_required=True
    )
    if p_ok and isinstance(positions, list):
        active_pos = [p for p in positions if abs(float(p.get("size", 0.0))) > 0]
        print(f"\n✅ Active Margined Futures Positions: {len(active_pos)}")
        for pos in active_pos:
            print(f"   Symbol: {pos.get('product_symbol')} | Size: {pos.get('size')} | Entry: {pos.get('entry_price')} | Unrealized PnL: {pos.get('unrealized_pnl')}")
        if not active_pos:
            print("   (No currently open futures positions)")
    else:
        print(f"⚠️  Positions query returned HTTP {p_status}: {positions}")

    # 3. Open Orders
    o_ok, orders, o_status = await verifier.request(
        "GET", "/v2/orders", params={"contract_types": "perpetual_futures,futures"}, auth_required=True
    )
    if o_ok and isinstance(orders, list):
        print(f"\n✅ Open Futures Orders: {len(orders)}")
        for o in orders:
            print(f"   ID: {o.get('id')} | Symbol: {o.get('product_symbol')} | Side: {o.get('side')} | Price: {o.get('limit_price')}")
        if not orders:
            print("   (No currently open orders)")
    else:
        print(f"⚠️  Open orders query returned HTTP {o_status}: {orders}")
        
    return True


async def test_order_execution_and_cancellation(verifier: DeltaVerifier, found_assets: Dict[str, Dict[str, Any]], symbol: str = "BTCUSD"):
    """Safely test order placement (deep out-of-the-money) and immediate cancellation."""
    print("\n" + "=" * 70)
    print(f"STEP 4: Testing Live Safe Bracket Order Placement & Cancellation ({symbol})")
    print("=" * 70)
    
    if not verifier.api_key or not verifier.api_secret:
        print("⚠️  SKIPPED: Credentials required for live order execution test.")
        return
        
    asset = found_assets.get(symbol)
    if not asset:
        print(f"❌ Asset {symbol} not found in verified universe.")
        return
        
    pid = asset["product_id"]
    tick_size = asset["tick_size"]
    mark_price = asset["mark_price"]
    
    if mark_price <= 0:
        print(f"❌ Mark price for {symbol} is 0. Cannot calculate safe limit price.")
        return
        
    # Safe limit price: 50% below mark price (guaranteed not to fill immediately)
    safe_entry = round_to_tick(mark_price * 0.50, tick_size, 'DOWN')
    safe_sl = round_to_tick(safe_entry * 0.95, tick_size, 'DOWN')
    safe_tp = round_to_tick(safe_entry * 1.10, tick_size, 'UP')
    
    order_body = {
        "product_id": pid,
        "side": "buy",
        "size": 1,  # 1 contract
        "order_type": "limit_order",
        "limit_price": format_price(safe_entry, tick_size),
        "bracket_stop_loss_price": format_price(safe_sl, tick_size),
        "bracket_take_profit_price": format_price(safe_tp, tick_size),
        "client_order_id": f"verify_test_{int(time.time())}"
    }
    
    print(f"Placing SAFE limit buy order (50% below mark ${mark_price:,.2f}):")
    print(f"   Product ID: {pid} ({symbol})")
    print(f"   Limit Price: {order_body['limit_price']}")
    print(f"   Bracket SL : {order_body['bracket_stop_loss_price']}")
    print(f"   Bracket TP : {order_body['bracket_take_profit_price']}")
    print(f"   Client ID  : {order_body['client_order_id']}")
    
    ok, res, status = await verifier.request("POST", "/v2/orders", body=order_body, auth_required=True)
    
    if not ok:
        print(f"❌ Order placement rejected: HTTP {status}, response: {res}")
        return
        
    order_id = res.get("id")
    print(f"\n🎉 ORDER PLACED SUCCESSFULLY! Order ID: {order_id}, State: {res.get('state')}")
    
    # Wait 1 second
    await asyncio.sleep(1)
    
    # Immediately Cancel Order
    print(f"\nCancelling test order {order_id}...")
    c_ok, c_res, c_status = await verifier.request(
        "DELETE", "/v2/orders", body={"id": order_id, "product_id": pid}, auth_required=True
    )
    
    if c_ok:
        print(f"✅ ORDER CANCELLED SUCCESSFULLY! Confirmed cancellation on exchange.")
    else:
        print(f"⚠️  Order cancellation returned HTTP {c_status}: {c_res}")
        # Try cancel all for safety
        await verifier.request("DELETE", "/v2/orders/all", body={"product_id": pid}, auth_required=True)
        print("   Dispatched fallback cancel_all for product.")


async def main():
    parser = argparse.ArgumentParser(description="Delta Exchange API & Futures Execution Verification")
    parser.add_argument("--api-key", default=os.getenv("DELTA_API_KEY", ""), help="Delta API Key")
    parser.add_argument("--api-secret", default=os.getenv("DELTA_API_SECRET", ""), help="Delta API Secret")
    parser.add_argument("--base-url", default=os.getenv("DELTA_API_URL", "https://api.india.delta.exchange"), help="Delta API Base URL")
    parser.add_argument("--dry-run", action="store_true", help="Run only public futures and rounding checks (no private API)")
    parser.add_argument("--test-order", action="store_true", help="Place and immediately cancel a safe test bracket order")
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to use for test order (default: BTCUSD)")
    args = parser.parse_args()

    api_key = args.api_key.strip()
    api_secret = args.api_secret.strip()
    
    print("\n" + "=" * 70)
    print("DELTA EXCHANGE INDIA — FUTURES EXECUTION & API VERIFIER")
    print("=" * 70)
    print(f"Base URL    : {args.base_url}")
    print(f"API Key     : {mask_secret(api_key)}")
    print(f"API Secret  : {mask_secret(api_secret)}")
    print(f"Dry Run Mode: {args.dry_run}")
    print(f"Test Order  : {args.test_order}")
    print("=" * 70)

    verifier = DeltaVerifier(api_key, api_secret, args.base_url)
    
    try:
        # Step 1: Verify Public Futures Universe
        found_assets = await verify_public_futures(verifier)
        
        # Step 2: Verify Tick-Size Rounding for SL/TP
        if found_assets:
            verify_rounding_and_sl_tp(found_assets)
            
        # Step 3: Verify Private Futures Wallet & Positions (if credentials available & not dry run)
        if not args.dry_run and api_key and api_secret:
            auth_ok = await verify_private_wallet_and_trading(verifier)
            
            # Step 4: Optional Order Execution & Cancellation test
            if auth_ok and args.test_order:
                await test_order_execution_and_cancellation(verifier, found_assets, symbol=args.symbol)
        elif not args.dry_run and (not api_key or not api_secret):
            print("\n💡 NOTE: To verify your live futures wallet, set DELTA_API_KEY and DELTA_API_SECRET in .env")
            print("   or run: python scripts/verify_delta_execution.py --api-key <KEY> --api-secret <SECRET>")
            
        print("\n" + "=" * 70)
        print("VERIFICATION COMPLETED")
        print("=" * 70 + "\n")
        
    finally:
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())
