#!/usr/bin/env python3
"""
SHAP Analizi Modülü
Global ve lokal SHAP açıklamaları oluşturur.
"""

import shap
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path


def shap_summary_png(model, x_sample: pd.DataFrame, out_path: str = "reports/shap_summary.png"):
    """
    Modelin genel (global) SHAP değerlerini hesaplar ve summary plot olarak PNG kaydeder.
    
    Args:
        model: Eğitilmiş model
        x_sample: Örnek veri (DataFrame)
        out_path: Çıktı PNG dosyası yolu
    """
    try:
        # Çıktı klasörünü oluştur
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        print("🔍 SHAP summary analizi başlıyor...")
        print(f"📊 Örnek veri boyutu: {x_sample.shape}")
        
        # SHAP explainer oluştur
        explainer = shap.Explainer(model)
        shap_values = explainer(x_sample)
        
        # Summary plot oluştur
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, x_sample, show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ SHAP summary grafiği kaydedildi: {out_path}")
        return out_path
        
    except (ValueError, RuntimeError, IOError) as e:
        raise RuntimeError(f"SHAP summary grafiği oluşturulamadı: {str(e)}") from e


def shap_local_png(model, x_one: pd.DataFrame, out_path: str = "reports/shap_local.png"):
    """
    Tek bir örnek (lokal) için SHAP waterfall grafiği oluşturur.
    
    Args:
        model: Eğitilmiş model
        x_one: Tek satırlık DataFrame
        out_path: Çıktı PNG dosyası yolu
    """
    try:
        # Çıktı klasörünü oluştur
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        print("🔍 SHAP lokal analizi başlıyor...")
        print(f"🎯 Örnek veri boyutu: {x_one.shape}")
        
        # SHAP explainer oluştur
        explainer = shap.Explainer(model)
        shap_values = explainer(x_one)
        
        # Waterfall plot oluştur
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ SHAP lokal grafiği kaydedildi: {out_path}")
        return out_path
        
    except (ValueError, RuntimeError, IOError) as e:
        raise RuntimeError(f"SHAP lokal grafiği oluşturulamadı: {str(e)}") from e


def load_sample_data(csv_path: str = "train_rul.csv", features_path: str = "selected_features.txt", sample_size: int = 300):
    """
    Eğitim verisinden örnek veri yükle
    
    Args:
        csv_path: Eğitim verisi CSV dosyası
        features_path: Seçili özellikler dosyası
        sample_size: Örnek boyutu
    
    Returns:
        pd.DataFrame: Örnek veri
    """
    try:
        # Özellikleri oku
        with open(features_path, "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        # Veriyi yükle
        df = pd.read_csv(csv_path)
        
        # Sadece seçili özellikleri al
        X = df[features]
        
        # Örnek al
        if len(X) > sample_size:
            x_sample = X.sample(n=sample_size, random_state=42)
        else:
            x_sample = X
            
        print(f"📊 Örnek veri yüklendi: {x_sample.shape}")
        return x_sample
        
    except (FileNotFoundError, ValueError, IOError) as e:
        raise FileNotFoundError(f"Örnek veri yüklenemedi: {str(e)}") from e


def test_shap_functions():
    """SHAP fonksiyonlarını test et"""
    try:
        print("🧪 SHAP fonksiyonları test ediliyor...")
        
        # Model ve scaler yükle
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # Örnek veri yükle
        x_sample = load_sample_data(sample_size=100)  # Test için küçük örnek
        
        # Veriyi ölçekle
        x_scaled = pd.DataFrame(
            scaler.transform(x_sample), 
            columns=x_sample.columns,
            index=x_sample.index
        )
        
        # SHAP summary test
        summary_path = shap_summary_png(model, x_scaled)
        
        # SHAP lokal test (ilk satır)
        x_one = x_scaled.iloc[[0]]  # İlk satırı al
        local_path = shap_local_png(model, x_one)
        
        print("🎉 SHAP testleri başarılı!")
        print(f"📊 Summary: {summary_path}")
        print(f"🎯 Local: {local_path}")
        
        return True
        
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"❌ SHAP testleri başarısız: {e}")
        return False


if __name__ == '__main__':
    # Test çalıştır
    success = test_shap_functions()
    
    if success:
        print("\n✅ SHAP modülü hazır!")
    else:
        print("\n💥 SHAP modülü test edilemedi!")