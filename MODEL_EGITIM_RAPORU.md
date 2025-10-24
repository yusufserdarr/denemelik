# Model Eğitim Raporu

## Tarih: 23 Ekim 2025

### Veri Seti
- **Dosya:** train_rul.csv
- **Toplam Kayıt:** 20,633 satır
- **Özellikler:** 10 sensör ölçümü (selected_features.txt)
- **Hedef Değişken:** RUL (Remaining Useful Life - Kalan Faydalı Ömür)

---

## Eğitilen Modeller

### 1. Klasik Modeller (Quantile Regression)

#### XGBoost
- **Modeller:** 
  - xgboost_q10_model.pkl (10. yüzdelik)
  - xgboost_q50_model.pkl (medyan)
  - xgboost_q90_model.pkl (90. yüzdelik)
- **Scaler:** xgboost_scaler.pkl
- **Performans (Medyan - q50):**
  - MAE: 25.99
  - RMSE: 35.82
  - R²: 0.702

#### LightGBM
- **Modeller:** 
  - lightgbm_q10_model.pkl
  - lightgbm_q50_model.pkl
  - lightgbm_q90_model.pkl
- **Scaler:** lightgbm_scaler.pkl
- **Performans (Medyan - q50):**
  - MAE: 25.67
  - RMSE: 35.53
  - R²: 0.707

#### CatBoost
- **Modeller:** 
  - catboost_q10_model.pkl
  - catboost_q50_model.pkl
  - catboost_q90_model.pkl
- **Scaler:** catboost_scaler.pkl
- **Performans (Medyan - q50):**
  - MAE: 26.91
  - RMSE: 36.81
  - R²: 0.686

---

### 2. LSTM Modeli (Derin Öğrenme)

- **Modeller:** 
  - lstm_q10_model.h5
  - lstm_q50_model.h5
  - lstm_q90_model.h5
- **Scaler:** lstm_scaler.pkl
- **Sekans Uzunluğu:** 50 zaman adımı
- **Performans:**
  - **q10:** MAE: 21.52, RMSE: 30.63, R²: 0.670
  - **q50:** MAE: 15.88, RMSE: 23.90, R²: 0.799 ⭐ (En İyi Performans)
  - **q90:** MAE: 30.07, RMSE: 43.25, R²: 0.341

---

### 3. Stacking Ensemble

- **Model:** model_stack.pkl
- **Meta Bilgiler:** model_stack_meta.json
- **Temel Modeller:** XGBoost, LightGBM, CatBoost
- **Meta-Öğrenici:** Linear Regression
- **Performans:**
  - MAE: 25.70
  - RMSE: 35.19
  - R²: 0.713

---

### 4. Survival Analysis (Cox Proportional Hazards)

- **Model:** cox_model.pkl
- **Scaler:** scaler_cox.pkl
- **Performans:**
  - Concordance Index: 0.9944 ⭐ (Mükemmel)

---

### 5. Conformal Prediction (Güven Aralıkları)

- **Model:** conformal_model.pkl
- **Meta Bilgiler:** conformal_meta.json
- **Quantile'lar:** 0.05, 0.10, 0.50, 0.90, 0.95
- **Performans:**
  - %90 Coverage: 0.909 (Hedef: 0.90) ✓
  - Ortalama Aralık Genişliği: 213.59 döngü
  - RMSE (p50): 47.48

---

## Özet

✅ **Tüm 5 model türü başarıyla eğitildi:**

1. ✅ Klasik Modeller (XGBoost, LightGBM, CatBoost) - 9 model dosyası
2. ✅ LSTM Modeli - 3 model dosyası
3. ✅ Stacking Ensemble - 1 model dosyası
4. ✅ Survival (Cox) Modeli - 1 model dosyası
5. ✅ Conformal Prediction - 1 model dosyası

**Toplam:** 15 model dosyası + 7 scaler/meta dosyası

---

## En İyi Performans Gösteren Modeller

1. **Survival (Cox):** Concordance Index: 0.9944
2. **LSTM (q50):** R²: 0.799, MAE: 15.88
3. **Stacking:** R²: 0.713, MAE: 25.70

---

## Kullanım

Her model türü farklı amaçlar için kullanılabilir:

- **Klasik Modeller:** Hızlı tahmin, açıklanabilirlik
- **LSTM:** Zaman serisi desenleri, yüksek doğruluk
- **Stacking:** Ensemble performansı
- **Survival:** Hayatta kalma analizi, olay tahmini
- **Conformal:** Güven aralıkları, belirsizlik ölçümü

---

## Gerekli Kütüphaneler

Tüm modeller requirements.txt dosyasındaki kütüphanelerle çalışır:
- xgboost, lightgbm, catboost
- tensorflow (LSTM için)
- lifelines (Survival için)
- scikit-learn (Conformal için)


