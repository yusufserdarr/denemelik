#!/usr/bin/env python3
"""
LIME Açıklama Modülü
Tabular veriler için LIME açıklamaları oluşturur.
"""

import pandas as pd
import numpy as np
import joblib
from lime import lime_tabular
from pathlib import Path
import webbrowser
import os


def explain_instance(model, scaler, x_orig_df_row, feature_names, out_html="reports/lime_ex.html"):
    """
    Tek örnek için LIME açıklaması oluştur ve HTML olarak kaydet
    
    Args:
        model: Eğitilmiş model (XGBoost vs.)
        scaler: StandardScaler objesi
        x_orig_df_row: Orijinal ölçekteki DataFrame satırı (1 satır)
        feature_names: Özellik isimleri listesi
        out_html: Çıktı HTML dosyası yolu
    
    Returns:
        str: Oluşturulan HTML dosyasının yolu
    """
    try:
        # Çıktı klasörünü oluştur
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        
        # Eğitim verisini yükle (LIME için referans veri gerekli)
        train_data = pd.read_csv("train_rul.csv")
        x_train_orig = train_data[feature_names]
        
        # Eğitim verisini ölçekle
        x_train_scaled = scaler.transform(x_train_orig)
        
        # Test verisini ölçekle
        x_scaled = scaler.transform(x_orig_df_row)
        
        print("🔍 LIME açıklaması oluşturuluyor...")
        print(f"📊 Eğitim veri boyutu: {x_train_scaled.shape}")
        print(f"🎯 Test veri boyutu: {x_scaled.shape}")
        
        # LIME Tabular Explainer oluştur
        explainer = lime_tabular.LimeTabularExplainer(
            x_train_scaled,
            feature_names=feature_names,
            class_names=['RUL'],
            mode='regression',
            discretize_continuous=True,
            random_state=42
        )
        
        # Açıklamayı oluştur
        explanation = explainer.explain_instance(
            x_scaled[0],  # İlk (ve tek) satır
            model.predict,
            num_features=len(feature_names),
            num_samples=1000
        )
        
        # HTML'e kaydet
        explanation.save_to_file(out_html)
        
        print(f"✅ LIME açıklaması oluşturuldu: {out_html}")
        
        return out_html
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"LIME açıklaması oluşturulamadı: {str(e)}") from e


def open_html_in_browser(html_path):
    """HTML dosyasını varsayılan tarayıcıda aç"""
    try:
        # Mutlak yol al
        abs_path = os.path.abspath(html_path)
        
        # Tarayıcıda aç
        webbrowser.open(f'file://{abs_path}')
        print(f"🌐 HTML dosyası tarayıcıda açıldı: {abs_path}")
        
    except (OSError, webbrowser.Error) as e:
        print(f"⚠️ HTML dosyası açılamadı: {e}")


def test_lime_explanation():
    """LIME açıklamasını test et"""
    try:
        print("🧪 LIME açıklaması test ediliyor...")
        
        # Model ve scaler yükle
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # Özellikleri oku
        with open("selected_features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        # Test verisi oluştur (örnek değerler)
        test_data = {
            'sensor_measurement_11': [47.5],
            'sensor_measurement_12': [521.0],
            'sensor_measurement_4': [1400.0],
            'sensor_measurement_7': [553.0],
            'sensor_measurement_15': [8.4],
            'sensor_measurement_9': [9050.0],
            'sensor_measurement_21': [23.3],
            'sensor_measurement_20': [38.9],
            'sensor_measurement_2': [642.0],
            'sensor_measurement_3': [1585.0]
        }
        
        test_df = pd.DataFrame(test_data)
        
        # LIME açıklaması oluştur
        html_file = explain_instance(model, scaler, test_df, features)
        
        # Tarayıcıda aç
        open_html_in_browser(html_file)
        
        print("🎉 LIME test başarılı!")
        return True
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"❌ LIME test başarısız: {e}")
        return False


if __name__ == '__main__':
    # Test çalıştır
    success = test_lime_explanation()
    
    if success:
        print("\n✅ LIME modülü hazır!")
    else:
        print("\n💥 LIME modülü test edilemedi!")
