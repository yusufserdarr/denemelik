#!/usr/bin/env python3
"""
Bakım Analizi Aracı
CSV dosyasından RUL değerlerini okuyup bakım kararları verir.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from maintenance import maintenance_decision


def analyze_maintenance_from_csv(csv_file, rul_column="RUL", output_file=None):
    """CSV dosyasından bakım analizi yap"""
    
    try:
        # CSV dosyasını oku
        df = pd.read_csv(csv_file)
        
        if rul_column not in df.columns:
            print(f"Hata: '{rul_column}' sütunu bulunamadı!")
            print(f"Mevcut sütunlar: {list(df.columns)}")
            return False
        
        # Her RUL değeri için bakım kararı ver
        results = []
        for index, row in df.iterrows():
            rul_value = row[rul_column]
            decision = maintenance_decision(rul_value)
            
            result = {
                'index': index,
                'rul': rul_value,
                'status': decision['status'],
                'message': decision['message'],
                'color': decision['color']
            }
            results.append(result)
        
        # Sonuçları DataFrame'e çevir
        results_df = pd.DataFrame(results)
        
        # İstatistikleri hesapla
        status_counts = results_df['status'].value_counts()
        total_count = len(results_df)
        
        print(f"📊 Bakım Analizi Sonuçları ({csv_file})")
        print("=" * 60)
        print(f"Toplam kayıt sayısı: {total_count}")
        print("\nDurum dağılımı:")
        for status, count in status_counts.items():
            percentage = (count / total_count) * 100
            print(f"  {status:8}: {count:4d} ({percentage:5.1f}%)")
        
        # Kritik durumları listele
        critical_cases = results_df[results_df['status'] == 'CRITICAL']
        if not critical_cases.empty:
            print(f"\n🚨 Acil bakım gereken kayıtlar ({len(critical_cases)} adet):")
            for _, case in critical_cases.head(10).iterrows():
                print(f"  Index {case['index']:3d}: RUL = {case['rul']:6.1f}")
            if len(critical_cases) > 10:
                print(f"  ... ve {len(critical_cases) - 10} adet daha")
        
        # Dosyaya kaydet
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Sonuçlar kaydedildi: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Hata: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='CSV dosyasından bakım analizi yapar',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python maintenance_analyzer.py data.csv
  python maintenance_analyzer.py data.csv --rul-column remaining_life
  python maintenance_analyzer.py data.csv --output results.csv
        """
    )
    
    parser.add_argument('csv_file', help='Analiz edilecek CSV dosyası')
    parser.add_argument('--rul-column', default='RUL', help='RUL değerlerinin bulunduğu sütun adı (default: RUL)')
    parser.add_argument('--output', help='Sonuçların kaydedileceği CSV dosyası')
    
    args = parser.parse_args()
    
    # Dosya var mı kontrol et
    if not Path(args.csv_file).exists():
        print(f"Hata: '{args.csv_file}' dosyası bulunamadı!")
        return 1
    
    # Analizi çalıştır
    success = analyze_maintenance_from_csv(args.csv_file, args.rul_column, args.output)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
