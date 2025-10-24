#!/usr/bin/env python3
"""
drift_detector.py

Model Drift Detection - Gerçek Zamanlı Model İzleme
PSI (Population Stability Index) ve ADWIN (Adaptive Windowing) algoritmaları ile
veri dağılımı değişimini tespit eder ve otomatik yeniden eğitim tetikler.

Teorik Temel:
- PSI: İki veri seti arasındaki dağılım farkını ölçer
- ADWIN: Zaman serisi verilerinde drift'i gerçek zamanlı tespit eder
- Model Performance Drift: Tahmin performansındaki düşüşü izler

Çıktılar:
- drift_alerts.json : Drift uyarıları ve metrikleri
- drift_history.csv : Drift geçmişi
- retrain_trigger.json : Yeniden eğitim tetikleyicisi
"""

import os
import json
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib  # type: ignore

# Drift detection imports
try:
    from scipy import stats  # type: ignore
    print("Scipy drift detector yuklendi")
except ImportError:
    print("scipy yuklu degil: pip install scipy")
    stats = None

from constants import FilePaths, ColumnNames

# =======================================================
# Konfigürasyon
# =======================================================
DRIFT_THRESHOLD_PSI = 0.2  # PSI eşiği (0.2 = orta drift)
DRIFT_THRESHOLD_ADWIN = 0.05  # ADWIN eşiği
PERFORMANCE_DECAY_THRESHOLD = 0.1  # Performans düşüşü eşiği (%10)
MIN_SAMPLES_FOR_DRIFT = 50  # Drift tespiti için minimum örnek sayısı
RETRAIN_COOLDOWN_HOURS = 24  # Yeniden eğitim arası minimum süre

# Dosya yolları
DRIFT_ALERTS_JSON = "drift_alerts.json"
DRIFT_HISTORY_CSV = "drift_history.csv"
RETRAIN_TRIGGER_JSON = "retrain_trigger.json"
REFERENCE_DATA_CSV = "reference_data.csv"

# =======================================================


class PSIDriftDetector:
    """Population Stability Index ile drift tespiti"""
    
    def __init__(self, threshold: float = DRIFT_THRESHOLD_PSI):
        self.threshold = threshold
        self.reference_data = None
        
    def fit_reference(self, data: pd.DataFrame, feature_cols: List[str]):
        """Referans veri setini kaydet"""
        self.reference_data = data[feature_cols].copy()
        self.feature_cols = feature_cols
        print(f"PSI referans verisi kaydedildi: {len(self.reference_data)} ornek")
        
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """PSI hesaplama"""
        # Bin sınırlarını belirle
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        
        if min_val == max_val:
            return 0.0
            
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Histogramları hesapla
        expected_hist, _ = np.histogram(expected, bins=bin_edges)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        
        # Sıfır değerleri önle
        expected_hist = expected_hist + 1e-6
        actual_hist = actual_hist + 1e-6
        
        # Oranları hesapla
        expected_ratio = expected_hist / np.sum(expected_hist)
        actual_ratio = actual_hist / np.sum(actual_hist)
        
        # PSI hesapla
        psi = np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio))
        return psi
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Drift tespiti yap"""
        if self.reference_data is None:
            return {"drift_detected": False, "error": "Referans veri yok"}
            
        results = {
            "drift_detected": False,
            "psi_scores": {},
            "max_psi": 0.0,
            "drift_features": []
        }
        
        for col in self.feature_cols:
            if col in current_data.columns:
                psi = self.calculate_psi(
                    self.reference_data[col].values,
                    current_data[col].values
                )
                results["psi_scores"][col] = psi
                results["max_psi"] = max(results["max_psi"], psi)
                
                if psi > self.threshold:
                    results["drift_detected"] = True
                    results["drift_features"].append(col)
                    
        return results


class StatisticalDriftDetector:
    """İstatistiksel drift tespiti (KS testi ve ortalama değişimi)"""
    
    def __init__(self, threshold: float = DRIFT_THRESHOLD_ADWIN):
        self.threshold = threshold
        self.feature_history = {}
        self.drift_history = []
        self.window_size = 100  # Son 100 değeri sakla
        
    def add_feature_value(self, feature_name: str, value: float):
        """Yeni değer ekle"""
        if feature_name not in self.feature_history:
            self.feature_history[feature_name] = []
            
        self.feature_history[feature_name].append(value)
        
        # Window boyutunu sınırla
        if len(self.feature_history[feature_name]) > self.window_size:
            self.feature_history[feature_name] = self.feature_history[feature_name][-self.window_size:]
            
    def detect_drift(self, feature_name: str) -> bool:
        """Drift tespiti yap"""
        if feature_name not in self.feature_history:
            return False
            
        values = self.feature_history[feature_name]
        if len(values) < 20:  # Yeterli veri yok
            return False
            
        # İlk yarı ve ikinci yarıyı karşılaştır
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        # Kolmogorov-Smirnov testi
        if stats is not None and len(first_half) > 5 and len(second_half) > 5:
            ks_stat, p_value = stats.ks_2samp(first_half, second_half)
            drift_detected = p_value < self.threshold
        else:
            # Basit ortalama değişimi kontrolü
            mean_first = np.mean(first_half)
            mean_second = np.mean(second_half)
            mean_change = abs(mean_second - mean_first) / (abs(mean_first) + 1e-6)
            drift_detected = mean_change > self.threshold
            
        if drift_detected:
            self.drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "feature": feature_name,
                "threshold": self.threshold
            })
            
        return drift_detected
        
    def get_drift_summary(self) -> Dict:
        """Drift özeti"""
        return {
            "total_drifts": len(self.drift_history),
            "recent_drifts": self.drift_history[-10:] if self.drift_history else [],
            "active_features": len(self.feature_history)
        }


class ModelPerformanceMonitor:
    """Model performans drift izleyicisi"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = []
        self.baseline_performance = None
        
    def update_performance(self, mae: float, rmse: float, r2: float):
        """Performans metriklerini güncelle"""
        performance = {
            "timestamp": datetime.now().isoformat(),
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        
        self.performance_history.append(performance)
        
        # Window boyutunu sınırla
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
            
        # Baseline performansı belirle
        if self.baseline_performance is None and len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            self.baseline_performance = {
                "mae": np.mean([p["mae"] for p in recent_performance]),
                "rmse": np.mean([p["rmse"] for p in recent_performance]),
                "r2": np.mean([p["r2"] for p in recent_performance])
            }
            
    def check_performance_drift(self) -> Dict:
        """Performans drift kontrolü"""
        if self.baseline_performance is None or len(self.performance_history) < 10:
            return {"drift_detected": False, "reason": "Yetersiz veri"}
            
        recent_performance = self.performance_history[-10:]
        current_mae = np.mean([p["mae"] for p in recent_performance])
        baseline_mae = self.baseline_performance["mae"]
        
        # Performans düşüşü hesapla
        performance_decay = (current_mae - baseline_mae) / baseline_mae
        
        drift_detected = performance_decay > PERFORMANCE_DECAY_THRESHOLD
        
        return {
            "drift_detected": drift_detected,
            "performance_decay": performance_decay,
            "current_mae": current_mae,
            "baseline_mae": baseline_mae,
            "threshold": PERFORMANCE_DECAY_THRESHOLD
        }


class DriftDetectorManager:
    """Ana drift detection yöneticisi"""
    
    def __init__(self):
        self.psi_detector = PSIDriftDetector()
        self.statistical_detector = StatisticalDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
        self.drift_alerts = []
        self.last_retrain_time = None
        
    def initialize_reference_data(self, data_path: str = REFERENCE_DATA_CSV):
        """Referans veri setini başlat"""
        if os.path.exists(data_path):
            ref_data = pd.read_csv(data_path)
            feature_cols = [col for col in ref_data.columns 
                           if col not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES, ColumnNames.RUL]]
            self.psi_detector.fit_reference(ref_data, feature_cols)
            print(f"Referans veri yuklendi: {data_path}")
        else:
            print(f"Referans veri bulunamadi: {data_path}")
            
    def process_live_data(self, data: pd.DataFrame) -> Dict:
        """Canlı veri ile drift kontrolü"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "psi_drift": {"drift_detected": False},
            "adwin_drift": {"drift_detected": False},
            "performance_drift": {"drift_detected": False},
            "overall_drift": False,
            "retrain_recommended": False
        }
        
        # PSI drift kontrolü
        if self.psi_detector.reference_data is not None:
            psi_result = self.psi_detector.detect_drift(data)
            results["psi_drift"] = psi_result
            
        # İstatistiksel drift kontrolü (her özellik için)
        statistical_drifts = []
        feature_cols = [col for col in data.columns 
                       if col not in [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES, ColumnNames.RUL]]
        
        for col in feature_cols:
            if col in data.columns and not data[col].isna().all():
                # Son değeri al
                last_value = data[col].dropna().iloc[-1] if not data[col].dropna().empty else 0
                self.statistical_detector.add_feature_value(col, last_value)
                drift_detected = self.statistical_detector.detect_drift(col)
                if drift_detected:
                    statistical_drifts.append(col)
                    
        results["statistical_drift"] = {
            "drift_detected": len(statistical_drifts) > 0,
            "drift_features": statistical_drifts
        }
        
        # Genel drift durumu
        results["overall_drift"] = (
            results["psi_drift"]["drift_detected"] or 
            results["statistical_drift"]["drift_detected"] or
            results["performance_drift"]["drift_detected"]
        )
        
        # Yeniden eğitim önerisi
        if results["overall_drift"]:
            results["retrain_recommended"] = self._should_retrain()
            
        # Uyarı kaydet
        if results["overall_drift"]:
            self._save_drift_alert(results)
            
        return results
        
    def update_model_performance(self, mae: float, rmse: float, r2: float):
        """Model performansını güncelle"""
        self.performance_monitor.update_performance(mae, rmse, r2)
        
        # Performans drift kontrolü
        perf_drift = self.performance_monitor.check_performance_drift()
        if perf_drift["drift_detected"]:
            print(f"Performans drift tespit edildi: {perf_drift['performance_decay']:.3f}")
            
    def _should_retrain(self) -> bool:
        """Yeniden eğitim gerekli mi?"""
        if self.last_retrain_time is None:
            return True
            
        time_since_retrain = datetime.now() - self.last_retrain_time
        return time_since_retrain.total_seconds() > (RETRAIN_COOLDOWN_HOURS * 3600)
        
    def _save_drift_alert(self, results: Dict):
        """Drift uyarısını kaydet"""
        alert = {
            "timestamp": results["timestamp"],
            "alert_type": "DRIFT_DETECTED",
            "details": results,
            "retrain_recommended": results["retrain_recommended"]
        }
        
        self.drift_alerts.append(alert)
        
        # Dosyaya kaydet
        with open(DRIFT_ALERTS_JSON, "w") as f:
            json.dump(self.drift_alerts[-50:], f, indent=2)  # Son 50 uyarı
            
        print(f"Drift uyarisi kaydedildi: {results['timestamp']}")
        
    def get_drift_summary(self) -> Dict:
        """Drift özeti"""
        return {
            "total_alerts": len(self.drift_alerts),
            "recent_alerts": self.drift_alerts[-5:] if self.drift_alerts else [],
            "statistical_summary": self.statistical_detector.get_drift_summary(),
            "performance_summary": self.performance_monitor.check_performance_drift(),
            "last_retrain": self.last_retrain_time.isoformat() if self.last_retrain_time else None
        }
        
    def trigger_retrain(self):
        """Yeniden eğitim tetikle"""
        self.last_retrain_time = datetime.now()
        
        retrain_info = {
            "timestamp": self.last_retrain_time.isoformat(),
            "triggered_by": "drift_detection",
            "status": "pending"
        }
        
        with open(RETRAIN_TRIGGER_JSON, "w") as f:
            json.dump(retrain_info, f, indent=2)
            
        print(f"Yeniden egitim tetiklendi: {self.last_retrain_time}")


def main():
    """Test fonksiyonu"""
    print("=== Model Drift Detection Test ===")
    
    # Drift detector'ı başlat
    detector_manager = DriftDetectorManager()
    
    # Referans veri yükle
    detector_manager.initialize_reference_data()
    
    # Test verisi oluştur
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sensor_measurement_11': np.random.normal(47.5, 5, 100),
        'sensor_measurement_12': np.random.normal(521.0, 50, 100),
        'sensor_measurement_4': np.random.normal(1400.0, 100, 100),
        'sensor_measurement_7': np.random.normal(553.0, 30, 100),
        'sensor_measurement_15': np.random.normal(8.4, 2, 100)
    })
    
    # Drift kontrolü
    results = detector_manager.process_live_data(test_data)
    print(f"Drift sonuclari: {results['overall_drift']}")
    
    # Özet
    summary = detector_manager.get_drift_summary()
    print(f"Drift ozeti: {summary}")
    
    print("=== Test Tamamlandi ===")


if __name__ == "__main__":
    main()
