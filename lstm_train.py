#!/usr/bin/env python3
"""
lstm_train.py - LSTM Model Eğitimi

Bu script, RUL tahmini için LSTM modellerini eğitir.
Quantile regresyon için üç farklı model eğitilir: q10, q50, q90
"""

import os
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

from constants import FilePaths, ColumnNames

# TensorFlow ve Keras
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

# Konfigürasyon
DATA_CSV = FilePaths.TRAIN_RUL_CSV
FEATURES_TXT = FilePaths.FEATURES_TXT
TEST_SIZE = 0.2
RANDOM_STATE = 42
SEQUENCE_LENGTH = 50  # Her sekans 50 zaman adımı içerecek
QUANTILES = [0.1, 0.5, 0.9]


def load_data():
    """Veriyi yükler ve özellik/hedef değişkenlerini hazırlar."""
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} dosyası bulunamadı.")
    
    df = pd.read_csv(DATA_CSV)

    with open(FEATURES_TXT, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    feature_cols = [f for f in selected_features if f not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES]]
    
    X = df[[ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES] + feature_cols]
    y = df[ColumnNames.RUL]
    return X, y, feature_cols


def create_sequences(X, y, sequence_length):
    """
    Unit bazında sekanslar oluşturur.
    Her bir unit için zaman serisi pencereleri oluşturulur.
    """
    sequences = []
    targets = []
    
    # Her unit için ayrı ayrı işle
    for unit in X[ColumnNames.UNIT_NUMBER].unique():
        unit_data = X[X[ColumnNames.UNIT_NUMBER] == unit].copy()
        unit_data = unit_data.sort_values(ColumnNames.TIME_IN_CYCLES)
        
        # Unit number ve time_in_cycles kolonlarını çıkar
        feature_cols = [col for col in unit_data.columns 
                       if col not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES]]
        unit_features = unit_data[feature_cols].values
        unit_targets = y[unit_data.index].values
        
        # Sekansları oluştur
        for i in range(len(unit_features) - sequence_length + 1):
            seq = unit_features[i:i + sequence_length]
            target = unit_targets[i + sequence_length - 1]
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)


def quantile_loss(q):
    """Quantile regression için kayıp fonksiyonu"""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        e = y_true - y_pred
        return tf.keras.backend.mean(tf.keras.backend.maximum(q * e, (q - 1) * e))
    return loss


def build_lstm_model(input_shape, quantile):
    """LSTM modeli oluşturur"""
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss=quantile_loss(quantile),
        metrics=['mae']
    )
    
    return model


def train_lstm_quantile(X_train_seq, y_train_seq, X_test_seq, y_test_seq, quantile, feature_cols):
    """Belirli bir quantile için LSTM modelini eğitir"""
    q_str = f"q{int(quantile*100)}"
    print(f"\n  > Quantile: {quantile} ({q_str}) icin LSTM modeli egitiliyor...")
    
    # Model oluştur
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_lstm_model(input_shape, quantile)
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Model eğitimi
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Tahmin
    y_pred = model.predict(X_test_seq, verbose=0).flatten()
    
    # Metrikler
    mae = mean_absolute_error(y_test_seq, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    r2 = r2_score(y_test_seq, y_pred)
    
    print(f"  {q_str}: MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
    
    # Modeli kaydet
    model_filename = f"lstm_{q_str}_model.h5"
    model.save(model_filename)
    print(f"  Model kaydedildi: {model_filename}")
    
    results = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, results, y_pred


def main():
    print("=== LSTM Model Egitimi ===\n")
    
    # Veriyi yükle
    X, y, feature_cols = load_data()
    
    # Veriyi 'unit_number' ve 'time_in_cycles' ile index'le
    X = X.set_index([ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES])
    y = y.to_frame().set_index(X.index)[ColumnNames.RUL]
    
    # Reset index - sekans oluşturma için gerekli
    X_reset = X.reset_index()
    y_reset = y.reset_index()[ColumnNames.RUL]
    
    # Unit bazlı train-test split
    unit_numbers = X.index.get_level_values(ColumnNames.UNIT_NUMBER).unique()
    train_units, test_units = train_test_split(unit_numbers, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_train_df = X_reset[X_reset[ColumnNames.UNIT_NUMBER].isin(train_units)]
    X_test_df = X_reset[X_reset[ColumnNames.UNIT_NUMBER].isin(test_units)]
    y_train_df = y_reset[X_reset[ColumnNames.UNIT_NUMBER].isin(train_units)]
    y_test_df = y_reset[X_reset[ColumnNames.UNIT_NUMBER].isin(test_units)]
    
    # Ölçekleme
    scaler = StandardScaler()
    X_train_scaled = X_train_df.copy()
    X_test_scaled = X_test_df.copy()
    
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train_df[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test_df[feature_cols])
    
    # Sekans oluştur
    print(f"Sekanslar olusturuluyor (sekans uzunlugu: {SEQUENCE_LENGTH})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_df, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_df, SEQUENCE_LENGTH)
    
    print(f"Train sekans sayisi: {len(X_train_seq)}")
    print(f"Test sekans sayisi: {len(X_test_seq)}")
    print(f"Sekans sekli: {X_train_seq.shape}")
    
    # Her quantile için model eğit
    all_results = {}
    
    for q in QUANTILES:
        model, results, y_pred = train_lstm_quantile(
            X_train_seq, y_train_seq, 
            X_test_seq, y_test_seq, 
            q, feature_cols
        )
        q_str = f"q{int(q*100)}"
        all_results[q_str] = results
    
    # Scaler'ı kaydet
    scaler_filename = "lstm_scaler.pkl"
    joblib.dump(scaler, scaler_filename)
    print(f"\nLSTM scaler kaydedildi: {scaler_filename}")
    
    print("\n=== LSTM Model Egitimi Tamamlandi ===")
    print("\nSonuclar:")
    for q_str, results in all_results.items():
        print(f"  {q_str}: MAE: {results['MAE']:.2f} | RMSE: {results['RMSE']:.2f} | R2: {results['R2']:.3f}")


if __name__ == "__main__":
    main()

