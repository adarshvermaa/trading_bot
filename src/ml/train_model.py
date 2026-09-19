import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def generate_synthetic_data(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic crypto 1m scalping training data."""
    np.random.seed(42)
    
    # Features:
    # 1. ema_cross: -1.0 (bearish), 0.0 (neutral), 1.0 (bullish)
    # 2. rsi / 100.0: normalized 0.0 to 1.0 (real 1m RSI typically 20 to 80)
    # 3. atr_norm: ATR / Close for 1m crypto is typically 0.0001 to 0.005 (0.01% - 0.5%)
    # 4. vwap_dist: (Close - VWAP) / VWAP typically -0.01 to +0.01
    # 5. vol_ratio: Current Vol / SMA(Vol, 20) typically 0.3 to 3.0
    # 6. adx / 100.0: normalized ADX (real 1m ADX typically 12 to 55)
    # 7. structure_score: 0.0 to 1.0 (ICT market structure confluence score)
    
    ema_cross = np.random.choice([-1.0, 0.0, 1.0], size=n_samples, p=[0.35, 0.30, 0.35])
    rsi = np.random.uniform(20.0, 80.0, size=n_samples)
    atr_norm = np.random.uniform(0.0001, 0.0035, size=n_samples)
    vwap_dist = np.random.normal(0.0, 0.003, size=n_samples)
    vol_ratio = np.random.uniform(0.4, 2.5, size=n_samples)
    adx = np.random.uniform(12.0, 55.0, size=n_samples)
    structure_score = np.random.uniform(0.1, 1.0, size=n_samples)
    
    X = np.column_stack([
        ema_cross, 
        rsi / 100.0,
        atr_norm, 
        vwap_dist, 
        vol_ratio, 
        adx / 100.0,
        structure_score
    ])
    
    y = np.zeros(n_samples, dtype=np.int64) # 0=NONE
    
    # 1. Trend Pullback Setups
    # LONG trend: Bullish EMA, healthy RSI (35-68), active momentum (ADX > 18), solid structure (>= 0.45), volume >= 0.7
    long_trend = (
        (ema_cross > 0) & 
        (rsi >= 35.0) & (rsi <= 68.0) & 
        (adx >= 18.0) & 
        (structure_score >= 0.45) & 
        (vol_ratio >= 0.70) &
        (vwap_dist >= -0.004)
    )
    # SHORT trend: Bearish EMA, healthy RSI (32-65), active momentum (ADX > 18), solid structure (>= 0.45), volume >= 0.7
    short_trend = (
        (ema_cross < 0) & 
        (rsi >= 32.0) & (rsi <= 65.0) & 
        (adx >= 18.0) & 
        (structure_score >= 0.45) & 
        (vol_ratio >= 0.70) &
        (vwap_dist <= 0.004)
    )
    
    # 2. Liquidity Sweep Reversal Setups (high structure_score >= 0.80, volume spike, at S/R extremes)
    # LONG sweep: Support sweep rejection (RSI <= 48, structure >= 0.80, vol >= 0.9)
    long_sweep = (
        (structure_score >= 0.80) & 
        (rsi <= 48.0) & 
        (vol_ratio >= 0.90) & 
        (vwap_dist <= 0.003)
    )
    # SHORT sweep: Resistance sweep rejection (RSI >= 52, structure >= 0.80, vol >= 0.9)
    short_sweep = (
        (structure_score >= 0.80) & 
        (rsi >= 52.0) & 
        (vol_ratio >= 0.90) & 
        (vwap_dist >= -0.003)
    )

    long_mask = (long_trend | long_sweep) & ~(short_trend | short_sweep)
    short_mask = (short_trend | short_sweep) & ~(long_trend | long_sweep)

    y[long_mask] = 1
    y[short_mask] = 2
    
    # Slight noise for generalization
    noise_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    y[noise_indices] = np.random.choice([0, 1, 2], size=len(noise_indices))
    
    return X.astype(np.float32), y

def main():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(10000)
    
    print(f"Data shape: {X.shape}, Label counts: {np.bincount(y)}")
    
    print("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    accuracy = model.score(X, y)
    print(f"Training accuracy: {accuracy:.4f}")
    
    print("Converting to ONNX...")
    initial_type = [('features', FloatTensorType([None, 7]))]
    onnx_model = convert_sklearn(model, 'scalper', initial_types=initial_type)
    
    # Create models directory at project root
    # Project root should be two levels up from src/ml
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'scalper_model.onnx')
    
    with open(model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model saved to {model_path}")
    
    # Verify
    print("Verifying ONNX model...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    test_feature = X[0:1]
    test_label = y[0]
    
    outputs = session.run(None, {input_name: test_feature})
    
    print(f"Test feature label (actual): {test_label}")
    print(f"ONNX predicted class: {outputs[0][0]}")
    print(f"ONNX probabilities: {outputs[1][0]}")
    print("Verification complete!")

if __name__ == '__main__':
    main()
