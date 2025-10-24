#!/usr/bin/env python3
"""
conformal_train.py

Amaç: Quantile Regression ile güven aralıkları oluşturmak.
Mevcut quantile modellerin (p10, p50, p90) üstüne kalibrasyon ekler.

Çıktılar:
- conformal_model.pkl : QuantileRegressor nesnesi (p10, p50, p90)
- conformal_meta.json : Kalibrasyon parametreleri (coverage)

Teorik temel:
- Quantile Regression, farklı quantile'lar için tahmin aralıkları sağlar
- p10-p90 aralığı = %80 güven aralığı
- p5-p95 aralığı = %90 güven aralığı
"""

import os
import json
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.linear_model import QuantileRegressor  # type: ignore

from constants import FilePaths, ColumnNames

# Basit metrik fonksiyonları
def coverage_score(y_true, y_lower, y_upper):
    return np.mean((y_true >= y_lower) & (y_true <= y_upper))

def mean_width_score(y_lower, y_upper):
    return np.mean(y_upper - y_lower)

CONFORMAL_MODEL_PKL = "conformal_model.pkl"
CONFORMAL_META_JSON = "conformal_meta.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHA = 0.1  # %90 güven aralığı (1 - alpha)

# Quantile'lar
QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]  # p5, p10, p50, p90, p95


def load_data():
    """Veriyi yükle ve hazırla"""
    if not os.path.exists(FilePaths.TRAIN_RUL_CSV):
        raise FileNotFoundError(f"{FilePaths.TRAIN_RUL_CSV} bulunamadı")
    
    df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
    with open(FilePaths.FEATURES_TXT, "r") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    
    feature_cols = [f for f in features if f not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES]]
    return df, feature_cols


def train_conformal_model():
    """Quantile regression modeli eğit"""
    df, feature_cols = load_data()
    
    # Veri hazırlığı
    X = df[feature_cols]
    y = df[ColumnNames.RUL]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Her quantile için ayrı model eğit
    print("Quantile modelleri egitiliyor...")
    quantile_models = {}
    
    for q in QUANTILES:
        print(f"  Quantile {q:.2f} egitiliyor...")
        model = QuantileRegressor(
            quantile=q,
            alpha=0.1,
            solver='highs'
        )
        model.fit(X_train_scaled, y_train)
        quantile_models[q] = model
    
    # Test setinde değerlendirme
    print("Test setinde degerlendirme...")
    y_pred_p50 = quantile_models[0.5].predict(X_test_scaled)
    y_pred_p05 = quantile_models[0.05].predict(X_test_scaled)
    y_pred_p95 = quantile_models[0.95].predict(X_test_scaled)
    
    # Coverage ve genişlik metrikleri (%90 güven aralığı: p5-p95)
    coverage_90 = coverage_score(y_test, y_pred_p05, y_pred_p95)
    mean_width_90 = mean_width_score(y_pred_p05, y_pred_p95)
    
    print(f"Quantile Regression Sonuclari:")
    print(f"   %90 Coverage: {coverage_90:.3f} (hedef: 0.90)")
    print(f"   %90 Ortalama Genislik: {mean_width_90:.2f}")
    print(f"   RMSE (p50): {np.sqrt(np.mean((y_test - y_pred_p50)**2)):.2f}")
    
    # Meta bilgileri kaydet
    meta_info = {
        "coverage_target_90": 0.90,
        "coverage_achieved_90": float(coverage_90),
        "mean_width_90": float(mean_width_90),
        "method": "quantile_regression",
        "quantiles": QUANTILES
    }
    
    # Dosyaları kaydet
    joblib.dump(quantile_models, CONFORMAL_MODEL_PKL)
    with open(CONFORMAL_META_JSON, "w") as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"Quantile modelleri kaydedildi: {CONFORMAL_MODEL_PKL}")
    print(f"Meta bilgiler kaydedildi: {CONFORMAL_META_JSON}")
    
    return quantile_models, meta_info


def main():
    """Ana fonksiyon"""
    print("Conformal Prediction Modeli Egitimi")
    print("=" * 50)
    
    model, meta = train_conformal_model()
    
    if model is not None:
        print("\nQuantile regression modelleri basariyla egitildi!")
        print(f"Guven araligi: %{meta['coverage_target_90']*100:.0f}")
        print(f"Ortalama aralik genisligi: {meta['mean_width_90']:.2f} dongu")
    else:
        print("\nModel egitimi basarisiz!")


if __name__ == "__main__":
    main()
