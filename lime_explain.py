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
        tuple: (HTML dosya yolu, LIME explanation objesi)
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
        
        # HTML dosyasını oku ve CSS ekle (dark mode uyumlu)
        with open(out_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Özel CSS ekle (okunaklı renkler)
        custom_css = """
        <style>
            body {
                background-color: #ffffff !important;
                color: #000000 !important;
                font-family: Arial, sans-serif;
                padding: 20px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #1f77b4 !important;
                font-weight: bold;
            }
            table {
                background-color: #ffffff !important;
                color: #000000 !important;
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            td, th {
                border: 1px solid #ddd !important;
                padding: 12px !important;
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            th {
                background-color: #f0f0f0 !important;
                font-weight: bold;
            }
            .positive {
                color: #2e7d32 !important;
                font-weight: bold;
            }
            .negative {
                color: #d32f2f !important;
                font-weight: bold;
            }
            div, span, p {
                color: #000000 !important;
            }
            /* Bar renklerini koru ama yazıları düzelt */
            svg text {
                fill: #000000 !important;
            }
        </style>
        """
        
        # <head> taginden hemen önce CSS ekle
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', f'<head>{custom_css}')
        else:
            # head yoksa body'den önce ekle
            html_content = custom_css + html_content
        
        # Güncellenmiş HTML'i kaydet
        with open(out_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ LIME açıklaması oluşturuldu (okunabilir renklerle): {out_html}")
        
        # HTML dosya yolu ve explanation objesini döndür
        return out_html, explanation
        
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
        html_file, _ = explain_instance(model, scaler, test_df, features)
        
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
