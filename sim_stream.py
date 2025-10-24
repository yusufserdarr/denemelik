#!/usr/bin/env python3
"""
Gerçek Zamanlı Sensör Simülatörü
Her saniye timestamp, sıcaklık, titreşim, tork verisi üretir.
"""

import argparse
import csv
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


class SensorSimulator:
    def __init__(self, output_file, interval, append_mode):
        self.output_file = Path(output_file)
        self.interval = interval
        self.append_mode = append_mode
        self.running = True
        self.row_count = 0
        self.csvfile = None
        self.writer = None
        
        # Loglama ayarla
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - INFO - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ctrl+C yakalama (platform bağımsız)
        signal.signal(signal.SIGINT, self.signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Ctrl+C yakalandığında nazikçe kapat"""
        self.logger.info(f"Durdurma sinyali alındı. Toplam {self.row_count} satır yazıldı.")
        self.running = False
        
    def setup_output_file(self):
        """Çıktı dosyasını hazırla"""
        # Dizin yoksa oluştur (platform bağımsız)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosya var mı kontrol et
        file_exists = self.output_file.exists()
        
        if not file_exists:
            # Yeni dosya oluştur
            mode = 'w'
            write_header = True
            self.logger.info(f"Yeni dosya oluşturuluyor: {self.output_file}")
        elif self.append_mode:
            # Mevcut dosyaya ekle
            mode = 'a'
            write_header = False
            self.logger.info(f"Mevcut dosyaya ekleniyor: {self.output_file}")
        else:
            # Üzerine yaz
            mode = 'w'
            write_header = True
            self.logger.info(f"Dosya üzerine yazılıyor: {self.output_file}")
            
        return mode, write_header
        
    def generate_sensor_data(self):
        """Rastgele sensör verisi üret"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Milisaniye dahil
        sicaklik = round(random.uniform(200.0, 700.0), 2)
        titresim = round(random.uniform(0.1, 5.0), 3)
        tork = round(random.uniform(10.0, 100.0), 2)
        
        return {
            'timestamp': timestamp,
            'sicaklik': sicaklik,
            'titresim': titresim,
            'tork': tork
        }
        
    def run(self):
        """Simülatörü çalıştır"""
        try:
            mode, write_header = self.setup_output_file()
            
            # CSV dosyasını aç
            self.csvfile = open(self.output_file, mode, newline='', encoding='utf-8')
            fieldnames = ['timestamp', 'sicaklik', 'titresim', 'tork']
            self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames)
            
            # Başlık satırı yaz (gerekirse)
            if write_header:
                self.writer.writeheader()
                self.csvfile.flush()
                
            self.logger.info(f"Veri üretimi başladı (interval: {self.interval}s)")
            self.logger.info("Durdurmak için Ctrl+C basın")
            
            while self.running:
                try:
                    # Sensör verisi üret
                    data = self.generate_sensor_data()
                    
                    # CSV'ye yaz
                    self.writer.writerow(data)
                    self.csvfile.flush()  # Son satırı hemen diske yaz
                    
                    self.row_count += 1
                    
                    # Her 10 satırda bir log
                    if self.row_count % 10 == 0:
                        self.logger.info(f"Yazılan satır sayısı: {self.row_count}")
                    
                    # Bekleme
                    time.sleep(self.interval)
                    
                except KeyboardInterrupt:
                    # Ana döngüden çık
                    break
                    
        except Exception as e:
            self.logger.error(f"Hata oluştu: {e}")
            return 1
        finally:
            # Dosyayı güvenli şekilde kapat
            if self.csvfile:
                self.csvfile.flush()  # Son flush
                self.csvfile.close()
                
        self.logger.info(f"Simülatör durduruldu. Toplam {self.row_count} satır yazıldı.")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Gerçek zamanlı sensör simülatörü - Her saniye timestamp, sıcaklık, titreşim, tork verisi üretir',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python sim_stream.py                           # Default ayarlarla çalıştır
  python sim_stream.py --out logs/stream.csv     # Belirtilen dosyaya yaz
  python sim_stream.py --interval 0.5            # 0.5 saniye aralıklarla
  python sim_stream.py --append                  # Mevcut dosyaya ekle
        """
    )
    
    parser.add_argument(
        '--out', 
        default='logs/stream.csv',
        help='Çıktı dosyası yolu (default: logs/stream.csv)'
    )
    
    parser.add_argument(
        '--interval', 
        type=float, 
        default=1.0,
        help='Veri üretim aralığı (saniye) (default: 1.0)'
    )
    
    parser.add_argument(
        '--append', 
        action='store_true',
        help='Mevcut dosyaya ekle, yoksa üzerine yaz (default: False)'
    )
    
    args = parser.parse_args()
    
    # Parametre doğrulama
    if args.interval <= 0:
        print("Hata: --interval pozitif bir sayı olmalı", file=sys.stderr)
        return 1
    
    # Simülatörü başlat
    # Çıktı yolunu betik dizinine göre çöz (farklı çalışma dizinlerinde tutarlılık)
    script_dir = Path(__file__).resolve().parent
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (script_dir / out_path).resolve()

    simulator = SensorSimulator(out_path, args.interval, args.append)
    return simulator.run()


if __name__ == '__main__':
    sys.exit(main())