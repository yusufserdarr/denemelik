#!/usr/bin/env python3
"""
Proje Sabitleri
SonarQube string duplication kurallarına uyum için tüm sabit değerler
"""

# ========== BAKIM DURUMLARI ==========
class MaintenanceStatus:
    CRITICAL = "CRITICAL"
    PLANNED = "PLANNED"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"

# ========== RENKLER ==========
class Colors:
    CRITICAL = "#D32F2F"  # Kırmızı
    PLANNED = "#ED6C02"   # Turuncu
    NORMAL = "#2E7D32"    # Yeşil
    UNKNOWN = "#757575"   # Gri

# ========== MESAJLAR ==========
class Messages:
    CRITICAL_MAINTENANCE = "Acil bakım gerekli"
    PLANNED_MAINTENANCE = "Planlı bakım önerilir"
    NORMAL_OPERATION = "Normal çalışma"
    INVALID_RUL = "Geçersiz RUL değeri - kontrol gerekli"
    
    # GUI Mesajları
    MODEL_LOAD_ERROR = "Model Yükleme Hatası"
    MISSING_DATA = "Eksik Veri"
    INVALID_DATA = "Geçersiz Veri"
    PREDICTION_ERROR = "Tahmin Hatası"
    
    # Veri girişi mesajları
    ENTER_SENSOR_VALUES = "Lütfen sensör değerlerini girin."
    ENTER_VALID_NUMBERS = "Lütfen geçerli sayısal değerler girin."

# ========== DOSYA YOLLARI ==========
class FilePaths:
    MODEL_PKL = "model.pkl"
    SCALER_PKL = "scaler.pkl"
    FEATURES_TXT = "selected_features.txt"
    STREAM_CSV = "logs/stream.csv"
    TRAIN_RUL_CSV = "train_rul.csv"

# ========== VARSAYILAN EŞIKLER ==========
class DefaultThresholds:
    CRITICAL = 20
    PLANNED = 50

# ========== SÜTUN ADLARI ==========
class ColumnNames:
    UNIT_NUMBER = "unit_number"
    TIME_IN_CYCLES = "time_in_cycles"
    RUL = "RUL"
    SICAKLIK = "sicaklik"
    TITRESIM = "titresim"
    TORK = "tork"
    TIMESTAMP = "timestamp"
    STATUS = "status"
    MAX_CYCLE = "max_cycle"

# ========== HTTP METODLARI ==========
class HttpMethods:
    GET = "GET"
    POST = "POST"
    PUT = "PUT"

# ========== VERI KAYNAKLARI ==========
class DataSources:
    LIVE_STREAM = "Canlı Akış"
    FILE_UPLOAD = "Dosya Yükle"
    MANUAL_INPUT = "Manuel Giriş"

# ========== RAPOR SABİTLERİ ==========
class ReportConstants:
    RAW_DATA_SHEET = "Ham Kayıtlar"
    SUMMARY_SHEET = "Özet"
    STATISTICS_SHEET = "İstatistikler"
