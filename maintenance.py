#!/usr/bin/env python3
"""
Bakım Karar Sistemi
RUL (Remaining Useful Life) değerine göre bakım kararı verir.
"""

import math
from typing import Dict, Optional, Union


def maintenance_decision(rul: float, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Union[str, float]]:
    """
    RUL değerine göre bakım kararı verir.
    
    Args:
        rul (float): Kalan ömür değeri
        thresholds (dict, optional): Eşik değerleri {"critical": 20, "planned": 50}
    
    Returns:
        dict: {
            "rul": float,
            "status": str,
            "color": str,
            "message": str
        }
    """
    # Varsayılan eşikler
    if thresholds is None:
        thresholds = {
            "critical": 20,
            "planned": 50
        }
    
    # Geçersiz değer kontrolü
    if math.isnan(rul) or rul < 0:
        return {
            "rul": rul,
            "status": "UNKNOWN",
            "color": "#757575",
            "message": "Geçersiz RUL değeri - kontrol gerekli"
        }
    
    # Eşik değerleri al
    critical_threshold = thresholds.get("critical", 20)
    planned_threshold = thresholds.get("planned", 50)
    
    # Karar mantığı
    if rul < critical_threshold:
        return {
            "rul": rul,
            "status": "CRITICAL",
            "color": "#D32F2F",
            "message": "Acil bakım gerekli"
        }
    elif rul < planned_threshold:
        return {
            "rul": rul,
            "status": "PLANNED",
            "color": "#ED6C02",
            "message": "Planlı bakım önerilir"
        }
    else:
        return {
            "rul": rul,
            "status": "NORMAL",
            "color": "#2E7D32",
            "message": "Normal çalışma"
        }


def print_decision(rul: float, thresholds: Optional[Dict[str, float]] = None):
    """Bakım kararını güzel formatta yazdır"""
    decision = maintenance_decision(rul, thresholds)
    print(f"RUL: {decision['rul']:6.1f} → {decision['status']:8} | {decision['message']}")


if __name__ == "__main__":
    print("🔧 Bakım Karar Sistemi Test")
    print("=" * 50)
    
    # Test senaryoları
    test_cases = [
        15,   # CRITICAL
        35,   # PLANNED  
        80,   # NORMAL
        5,    # CRITICAL
        25,   # PLANNED
        100,  # NORMAL
        -5,   # UNKNOWN (negatif)
        float('nan')  # UNKNOWN (NaN)
    ]
    
    print("Varsayılan eşiklerle (critical: 20, planned: 50):")
    print("-" * 50)
    
    for rul in test_cases:
        print_decision(rul)
    
    print("\n" + "=" * 50)
    print("Özel eşiklerle (critical: 10, planned: 30):")
    print("-" * 50)
    
    custom_thresholds = {"critical": 10, "planned": 30}
    for rul in [5, 15, 25, 40]:
        print_decision(rul, custom_thresholds)
    
    print("\n" + "=" * 50)
    print("Kabul kriteri testleri:")
    print("-" * 50)
    
    # Kabul kriteri testleri
    test_results = [
        (15, "CRITICAL"),
        (35, "PLANNED"),
        (80, "NORMAL")
    ]
    
    all_passed = True
    for rul, expected_status in test_results:
        result = maintenance_decision(rul)
        actual_status = result["status"]
        passed = actual_status == expected_status
        all_passed = all_passed and passed
        
        status_icon = "✅" if passed else "❌"
        print(f"{status_icon} RUL {rul:2.0f} → Beklenen: {expected_status:8} | Gerçek: {actual_status:8}")
    
    print("-" * 50)
    if all_passed:
        print("🎉 Tüm testler başarılı!")
    else:
        print("⚠️  Bazı testler başarısız!")
