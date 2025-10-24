#!/usr/bin/env python3
"""
survival_train.py

Amaç: Cox Proportional Hazards (CoxPH) ile hayatta kalma analizi modeli eğitmek.
Çıktılar:
- cox_model.pkl : lifelines CoxPHFitter nesnesi (joblib ile)
- scaler_cox.pkl: Özellik ölçekleyici (StandardScaler)

Veri hazırlığı:
- RUL -> event_time (kalan ömür) ve event_observed (1: arıza, 0: sağ)
- Özellikler: selected_features.txt içindeki 10 sensör özelliği
"""

import os
import joblib  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from lifelines import CoxPHFitter  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from constants import FilePaths, ColumnNames


COX_MODEL_PKL = "cox_model.pkl"
SCALER_COX_PKL = "scaler_cox.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    if not os.path.exists(FilePaths.TRAIN_RUL_CSV):
        raise FileNotFoundError(f"{FilePaths.TRAIN_RUL_CSV} bulunamadı")
    df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
    with open(FilePaths.FEATURES_TXT, "r") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    feature_cols = [f for f in features if f not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES]]
    return df, feature_cols


def build_survival_frame(df: pd.DataFrame, feature_cols):
    # Cox modeli için: duration_col = event_time, event_col = event_observed
    # RUL: kalan döngü sayısı; arıza olduğunda RUL=0’a yaklaşır. Basit kurgu:
    # event_time = max_cycle - time_in_cycles, event_observed = 1 (arızaya kadar izlenenler)
    # Not: Gerçek üretim verilerinde sensör zamanı ve arıza etiketleriyle daha doğru kurulmalıdır.
    max_cycle_per_unit = df.groupby(ColumnNames.UNIT_NUMBER)[ColumnNames.TIME_IN_CYCLES].transform("max")
    event_time = (max_cycle_per_unit - df[ColumnNames.TIME_IN_CYCLES]).astype(float)
    # Arızaya ulaşan son kayıtları olay olarak işaretleyelim
    is_last = df[ColumnNames.TIME_IN_CYCLES] == max_cycle_per_unit
    event_observed = is_last.astype(int)

    X = df[feature_cols].copy()
    surv = pd.DataFrame({
        "event_time": event_time,
        "event_observed": event_observed
    }, index=df.index)
    return X, surv


def main():
    df, feature_cols = load_data()

    X, surv = build_survival_frame(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, surv, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=surv["event_observed"]
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    train_df = X_train_scaled.copy()
    train_df["event_time"] = y_train["event_time"]
    train_df["event_observed"] = y_train["event_observed"]

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="event_time", event_col="event_observed")

    # Basit doğrulama: kısmi olasılık
    concord = cph.concordance_index_
    print(f"CoxPH concordance_index: {concord:.4f}")

    joblib.dump(cph, COX_MODEL_PKL)
    joblib.dump(scaler, SCALER_COX_PKL)
    print(f"Cox model kaydedildi: {COX_MODEL_PKL}")
    print(f"Scaler kaydedildi: {SCALER_COX_PKL}")


if __name__ == "__main__":
    main()


