# 🔧 Predictive Maintenance System (Makine Öğrenimi ile Erken Arıza Tespiti)

Bu proje, lojistik ve endüstriyel sistemlerde kullanılan makine bileşenlerinin **arıza yapmadan önceki kalan ömürlerini (RUL – Remaining Useful Life)** tahmin ederek **kestirimci bakım planlaması** yapmayı amaçlar.  
Makine öğrenimi, veri akışı simülasyonu, model açıklanabilirliği (LIME & SHAP) ve dashboard arayüzleri bir arada kullanılmıştır.

---

## 🚀 Özellikler

### 🔹 Gerçek Zamanlı Veri Akışı
- `sim_stream.py` dosyası, sıcaklık, titreşim ve tork sensörlerinden gelen verileri simüle eder.  
- `logs/stream.csv` dosyasına her saniye yeni veri yazar.

### 🔹 Model Tahmini (XGBoost)
- Eğitilen model, her veri noktasının kalan ömrünü (RUL) tahmin eder.  
- `model.pkl` ve `scaler.pkl` dosyalarıyla tahmin süreci otomatik yapılır.

### 🔹 Bakım Planlama Modülü
RUL değerine göre bakım durumunu belirler:  
- 🟢 Normal  
- 🟠 Planlı bakım önerisi  
- 🔴 Acil bakım gerekli

### 🔹 Veri Kaydı ve Günlük Raporlama
- Her tahmin otomatik olarak `logs/predictions.csv` dosyasına kaydedilir.  
- Gün sonunda `reports/report_YYYY-MM-DD.xlsx` raporu oluşturulur (ham veriler + özet sayfası).

### 🔹 Model Açıklanabilirliği (Explainability)
- **LIME:** Tek bir örnek için modelin hangi sensörlerden etkilendiğini HTML grafiğiyle açıklar.  
- **SHAP:** Modelin genel (global) ve lokal kararlarını PNG grafikleriyle gösterir.

### 🔹 Çift Arayüz
- 🖥️ **PyQt5 GUI:** Masaüstü uygulaması  
- 🌐 **Streamlit Dashboard:** Web tabanlı izleme arayüzü

---


## ⚙️ Kurulum

```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
pandas, numpy, scikit-learn, xgboost, lime, shap, pyqt5, streamlit, matplotlib, openpyxl


## 📁 Proje Yapısı

.
├── app.py                   → Streamlit dashboard
├── main_gui.py              → PyQt5 masaüstü arayüzü
├── model_train.py           → Model eğitimi
├── maintenance.py           → Bakım planlama modülü
├── reporting.py             → Loglama ve Excel raporlama
├── sim_stream.py            → Gerçek zamanlı veri akışı simülasyonu
├── lime_explain.py          → LIME açıklamaları
├── shap_analysis.py         → SHAP açıklamaları
├── model.pkl                → Eğitilmiş model
├── scaler.pkl               → Veri ölçekleyici
├── logs/                    → Anlık veri akış kayıtları


