#!/usr/bin/env python3
"""
model_train.py - Gelişmiş Model Eğitimi ve Stacking

Bu script, XGBoost, LightGBM, CatBoost ve LSTM modellerini eğitir
ve stacking ensemble oluşturur.
"""

import os
import json
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

from constants import FilePaths, ColumnNames

# =======================================================
# Konfigürasyon
# =======================================================
DATA_CSV = FilePaths.TRAIN_RUL_CSV
FEATURES_TXT = FilePaths.FEATURES_TXT
TEST_SIZE = 0.2
RANDOM_STATE = 42
QUANTILES = [0.1, 0.5, 0.9] # Güven aralığı tahminleri
# Stacking meta-öğrenici çıktıları
STACK_MODEL_PKL = "model_stack.pkl"
STACK_META_INFO_JSON = "model_stack_meta.json"
# =======================================================


def load_data():
    """Veriyi yükler ve özellik/hedef değişkenlerini hazırlar."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} dosyası bulunamadı. Lütfen önce rul_generator.py'yi çalıştırın.")
    
    df = pd.read_csv(DATA_CSV)

    with open(FEATURES_TXT, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    # Unit number ve time_in_cycles'ı index yapmak için al
    feature_cols = [f for f in selected_features if f not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES]]
    
    # X, y ve feature_cols listesi döndürülür
    X = df[[ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES] + feature_cols]
    y = df[ColumnNames.RUL]
    return X, y, feature_cols


def train_and_evaluate_quantile(model_name, X_train, y_train, X_test, y_test, scaler, feature_cols):
    """Quantile regression modellerini eğitir ve değerlendirir."""
    print(f"--- {model_name} Ceyreklik Modelleri Egitiliyor ---")
    
    results = {}
    test_preds_p50 = None
    
    for q in QUANTILES:
        print(f"  > Quantile: {q} (q{int(q*100)}) icin model egitiliyor...")
        
        if model_name == "XGBoost":
            import xgboost as xgb  # type: ignore
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror',
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_name == "LightGBM":
            import lightgbm as lgb  # type: ignore
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=100,
                num_leaves=31,
                force_col_wise=True,
                verbosity=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_name == "CatBoost":
            import catboost as cb  # type: ignore
            model = cb.CatBoostRegressor(
                loss_function='Quantile',
                iterations=100,
                random_state=RANDOM_STATE,
                verbose=False
            )
        elif model_name == "LSTM":
            # LSTM için özel işlem
            model = None  # LSTM ayrı işlenecek
        else:
            continue
            
        if model is not None:
            # Model eğitimi
            model.fit(X_train[feature_cols], y_train)
            
            # Tahmin
            y_pred = model.predict(X_test[feature_cols])
            
            # Metrikler
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Sonuçları kaydet
            q_str = f"q{int(q*100)}"
            results[q_str] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
            # Modeli kaydet
            model_filename = f"{model_name.lower()}_{q_str}_model.pkl"
            joblib.dump(model, model_filename)
            
            # p50 tahminlerini sakla (stacking için)
            if q == 0.5:
                test_preds_p50 = y_pred
            
            print(f"  {q_str}: MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

    # Scaler'ı sadece bir kez kaydet
    scaler_filename = f"{model_name.lower()}_scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"{model_name} modelleri icin scaler {scaler_filename} olarak kaydedildi.")
    
    print(f"{model_name} Ceyreklik Modelleri Egitimi Tamamlandi.")
    
    return results, test_preds_p50


def main():
    X, y, feature_cols = load_data()
    
    # Veriyi 'unit_number' ve 'time_in_cycles' ile index'le (LSTM için gerekli)
    X = X.set_index([ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES])
    y = y.to_frame().set_index(X.index)[ColumnNames.RUL]

    # Rastgele birimleri eğitim ve test setlerine ayır
    unit_numbers = X.index.get_level_values(ColumnNames.UNIT_NUMBER).unique()
    train_units, test_units = train_test_split(unit_numbers, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_train_raw = X.loc[train_units]
    X_test_raw = X.loc[test_units]
    y_train_raw = y.loc[train_units]
    y_test_raw = y.loc[test_units]

    # Ölçekleme (Sadece sensör sütunlarını ölçekle)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw[feature_cols]),
        columns=feature_cols,
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw[feature_cols]),
        columns=feature_cols,
        index=X_test_raw.index
    )
    
    # Unit number ve time_in_cycles'ı geri ekle (LSTM'de sekanslama için gerekli)
    # MultiIndex seviyelerinden geri ekle
    X_train_scaled[ColumnNames.UNIT_NUMBER] = X_train_scaled.index.get_level_values(ColumnNames.UNIT_NUMBER)
    X_test_scaled[ColumnNames.UNIT_NUMBER] = X_test_scaled.index.get_level_values(ColumnNames.UNIT_NUMBER)
    
    # Tahminler için gerekli olan X_test ve y_test'i sakla
    X_test_eval = X_test_scaled[[ColumnNames.UNIT_NUMBER] + feature_cols]
    
    # Tüm modelleri eğit ve değerlendir
    all_results = {}
    test_meta_features = []  # p50 test tahminleri
    
    # XGBoost Quantile Regressor
    res, p50 = train_and_evaluate_quantile("XGBoost", X_train_scaled, y_train_raw, X_test_eval, y_test_raw, scaler, feature_cols)
    all_results["XGBoost"] = res
    if p50 is not None:
        test_meta_features.append(p50.reshape(-1, 1))

    # LightGBM Quantile Regressor
    res, p50 = train_and_evaluate_quantile("LightGBM", X_train_scaled, y_train_raw, X_test_eval, y_test_raw, scaler, feature_cols)
    all_results["LightGBM"] = res
    if p50 is not None:
        test_meta_features.append(p50.reshape(-1, 1))

    # CatBoost Quantile Regressor
    res, p50 = train_and_evaluate_quantile("CatBoost", X_train_scaled, y_train_raw, X_test_eval, y_test_raw, scaler, feature_cols)
    all_results["CatBoost"] = res
    if p50 is not None:
        test_meta_features.append(p50.reshape(-1, 1))

    # ---- Stacking Meta-Öğrenici ----
    if len(test_meta_features) >= 2:
        from sklearn.linear_model import LinearRegression  # type: ignore
        Z_test = np.hstack(test_meta_features)  # [n_test, n_base_models]
        meta = LinearRegression()
        meta.fit(Z_test, y_test_raw.values)
        y_stack = meta.predict(Z_test)
        mae = mean_absolute_error(y_test_raw, y_stack)
        rmse = float(np.sqrt(mean_squared_error(y_test_raw, y_stack)))
        r2 = r2_score(y_test_raw, y_stack)
        
        print(f"\n--- Stacking Meta-Ogrenici Sonuclari ---")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
        
        # Meta modeli kaydet
        joblib.dump(meta, STACK_MODEL_PKL)
        
        # Meta bilgileri kaydet
        meta_info = {
            "meta_model": "LinearRegression",
            "base_models": ["XGBoost", "LightGBM", "CatBoost"],
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_r2": float(r2),
            "n_base_models": len(test_meta_features)
        }
        
        with open(STACK_META_INFO_JSON, "w") as f:
            json.dump(meta_info, f, indent=2)
        
        print(f"Stacking meta modeli kaydedildi: {STACK_MODEL_PKL}")
        print(f"Meta bilgiler kaydedildi: {STACK_META_INFO_JSON}")
    else:
        print("Stacking icin yeterli taban model bulunamadi.")
    
    print("\n=== Model Egitimi Tamamlandi ===")


if __name__ == "__main__":
    main()
