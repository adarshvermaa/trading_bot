import os
import json
import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from src.ml.pattern_memory import FEATURE_NAMES, NUM_FEATURES, MarketPatternFingerprint


def generate_synthetic_data(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic crypto 1m scalping training data
    spanning all 20 pro-trader technical features.
    """
    np.random.seed(42)

    # 1. Momentum & Regime
    rsi_norm = np.random.uniform(0.20, 0.80, size=n_samples)
    adx_norm = np.random.uniform(0.12, 0.55, size=n_samples)
    atr_norm = np.random.uniform(0.0001, 0.0035, size=n_samples)
    vwap_dist = np.random.normal(0.0, 0.003, size=n_samples)
    vol_ratio = np.random.uniform(0.4, 2.8, size=n_samples)

    # 2. Trend & Squeeze
    ema_trend = np.random.choice([-1.0, 0.0, 1.0], size=n_samples, p=[0.35, 0.30, 0.35])
    is_squeeze = np.random.choice([0.0, 1.0], size=n_samples, p=[0.75, 0.25])

    # 3. Structure & Key Levels
    structure_score = np.random.uniform(0.1, 1.0, size=n_samples)
    dist_to_vah_pct = np.random.normal(0.002, 0.008, size=n_samples)
    dist_to_val_pct = np.random.normal(-0.002, 0.008, size=n_samples)
    dist_to_pdh_pct = np.random.normal(0.004, 0.010, size=n_samples)
    dist_to_pdl_pct = np.random.normal(-0.004, 0.010, size=n_samples)

    # 4. Trap & Price Action indicators
    is_bull_trap = np.random.choice([0.0, 1.0], size=n_samples, p=[0.90, 0.10])
    is_bear_trap = np.random.choice([0.0, 1.0], size=n_samples, p=[0.90, 0.10])
    is_judas_swing = np.random.choice([0.0, 1.0], size=n_samples, p=[0.92, 0.08])
    is_volume_absorption = np.random.choice([0.0, 1.0], size=n_samples, p=[0.90, 0.10])

    # 5. Candlestick Geometry
    upper_wick_ratio = np.random.uniform(0.05, 0.60, size=n_samples)
    lower_wick_ratio = np.random.uniform(0.05, 0.60, size=n_samples)
    body_ratio = 1.0 - (upper_wick_ratio + lower_wick_ratio)
    body_ratio = np.clip(body_ratio, 0.10, 0.85)

    # 6. Session & Cost
    spread_bps = np.random.uniform(0.5, 3.5, size=n_samples)

    X = np.column_stack([
        rsi_norm,
        adx_norm,
        atr_norm,
        vwap_dist,
        vol_ratio,
        ema_trend,
        is_squeeze,
        structure_score,
        dist_to_vah_pct,
        dist_to_val_pct,
        dist_to_pdh_pct,
        dist_to_pdl_pct,
        is_bull_trap,
        is_bear_trap,
        is_judas_swing,
        is_volume_absorption,
        upper_wick_ratio,
        lower_wick_ratio,
        body_ratio,
        spread_bps,
    ])

    y = np.zeros(n_samples, dtype=np.int64)  # 0=NONE, 1=LONG, 2=SHORT

    # LONG Setups:
    # A. Trend Pullback Long (Bullish EMA, VWAP bounce, solid structure, healthy RSI, normal spread)
    long_trend = (
        (ema_trend > 0) &
        (rsi_norm >= 0.35) & (rsi_norm <= 0.68) &
        (adx_norm >= 0.18) &
        (structure_score >= 0.50) &
        (vol_ratio >= 0.70) &
        (vwap_dist >= -0.003) &
        (spread_bps <= 2.5) &
        (is_bull_trap == 0.0)
    )

    # B. Bear Trap Reversal Long (Sell-side SFP below PDL/VAL, lower wick rejection >= 0.30, volume surge)
    long_trap = (
        (is_bear_trap == 1.0) &
        (lower_wick_ratio >= 0.30) &
        (vol_ratio >= 1.2) &
        (spread_bps <= 3.0)
    )

    # SHORT Setups:
    # A. Trend Pullback Short (Bearish EMA, VWAP rejection, solid structure, healthy RSI)
    short_trend = (
        (ema_trend < 0) &
        (rsi_norm >= 0.32) & (rsi_norm <= 0.65) &
        (adx_norm >= 0.18) &
        (structure_score >= 0.50) &
        (vol_ratio >= 0.70) &
        (vwap_dist <= 0.003) &
        (spread_bps <= 2.5) &
        (is_bear_trap == 0.0)
    )

    # B. Bull Trap Reversal Short (Buy-side SFP above PDH/VAH, upper wick rejection >= 0.30, volume surge)
    short_trap = (
        (is_bull_trap == 1.0) &
        (upper_wick_ratio >= 0.30) &
        (vol_ratio >= 1.2) &
        (spread_bps <= 3.0)
    )

    long_mask = (long_trend | long_trap) & ~(short_trend | short_trap)
    short_mask = (short_trend | short_trap) & ~(long_trend | long_trap)

    y[long_mask] = 1
    y[short_mask] = 2

    # Slight realistic noise (2%)
    noise_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    y[noise_indices] = np.random.choice([0, 1, 2], size=len(noise_indices))

    return X.astype(np.float32), y


def load_historical_pattern_data(patterns_dir: str = "data/patterns") -> tuple[np.ndarray, np.ndarray]:
    """Load real winning and losing patterns recorded by the bot."""
    win_path = os.path.join(patterns_dir, "winning_patterns.json")
    loss_path = os.path.join(patterns_dir, "losing_mistake_patterns.json")

    X_list = []
    y_list = []

    if os.path.exists(win_path):
        try:
            with open(win_path, "r") as f:
                wins = json.load(f)
                for w in wins:
                    fp = MarketPatternFingerprint.from_dict(w)
                    label = 1 if fp.direction.upper() == "LONG" else 2
                    X_list.append(fp.to_vector())
                    y_list.append(label)
        except Exception as e:
            print(f"Error loading winning patterns: {e}")

    if os.path.exists(loss_path):
        try:
            with open(loss_path, "r") as f:
                losses = json.load(f)
                for l in losses:
                    fp = MarketPatternFingerprint.from_dict(l)
                    # For losing trades, the label should be 0 (NONE) to teach the model to avoid
                    X_list.append(fp.to_vector())
                    y_list.append(0)
        except Exception as e:
            print(f"Error loading losing patterns: {e}")

    if X_list:
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)
    return np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int64)


def train_and_export_onnx(output_model_path: str = "models/scalper_model.onnx", from_history: bool = True):
    print("Generating baseline 20-feature dataset...")
    X_syn, y_syn = generate_synthetic_data(10000)

    X_hist, y_hist = load_historical_pattern_data()
    if len(y_hist) > 0 and from_history:
        print(f"Found {len(y_hist)} recorded historical patterns. Augmenting training set...")
        # Weight historical trade records heavily (10x replication)
        X_hist_aug = np.repeat(X_hist, 10, axis=0)
        y_hist_aug = np.repeat(y_hist, 10, axis=0)
        X = np.vstack([X_syn, X_hist_aug])
        y = np.concatenate([y_syn, y_hist_aug])
    else:
        X, y = X_syn, y_syn

    print(f"Final training set shape: {X.shape}, labels: {np.bincount(y)}")
    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"Model training accuracy: {accuracy:.4f}")

    print("Converting to ONNX format (20 input features)...")
    initial_type = [('features', FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_sklearn(model, 'scalper_20feats', initial_types=initial_type)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    with open(output_model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Successfully saved ONNX model to {output_model_path}")

    # Verification run
    print("Verifying ONNX model runtime inference...")
    session = ort.InferenceSession(output_model_path)
    input_name = session.get_inputs()[0].name
    test_feature = X[0:1]
    outputs = session.run(None, {input_name: test_feature})
    print(f"Test prediction: class={outputs[0][0]}, probabilities={outputs[1][0]}")
    print("ONNX verification successful!")


def main():
    parser = argparse.ArgumentParser(description="Train and export ONNX scalper model.")
    parser.add_argument("--output", type=str, default="models/scalper_model.onnx", help="Path to output ONNX model")
    parser.add_argument("--from-history", action="store_true", default=True, help="Incorporate real pattern history")
    args = parser.parse_args()

    train_and_export_onnx(output_model_path=args.output, from_history=args.from_history)


if __name__ == '__main__':
    main()
