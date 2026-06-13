# Laporan Proyek Machine Learning - Prediksi Customer Churn

## 1. Business Understanding

### Problem Statement
- Perusahaan telekomunikasi mengalami kehilangan pelanggan (churn) yang berdampak pada pendapatan.
- Bagaimana cara mengidentifikasi pelanggan yang berpotensi churn agar tim dapat melakukan tindakan retensi lebih awal?

### Goals
- Membangun model klasifikasi yang dapat memprediksi apakah seorang pelanggan akan churn atau tidak, berdasarkan data demografi, layanan yang digunakan, dan informasi tagihan.

### Solution Statement
- Mengimplementasikan dan membandingkan tiga algoritma klasifikasi: **Logistic Regression**, **Random Forest**, dan **XGBoost**.
- Model terbaik dipilih berdasarkan metrik F1-Score karena adanya ketidakseimbangan kelas pada data churn.

---

## 2. Data Understanding

Dataset yang digunakan adalah **Telco Customer Churn** dari Kaggle, terdiri dari [ISI: jumlah baris] baris dan [ISI: jumlah kolom] kolom.

### Fitur pada Dataset
| Fitur | Deskripsi |
|---|---|
| customerID | ID unik pelanggan (dihapus saat preprocessing) |
| gender | Jenis kelamin pelanggan |
| SeniorCitizen | Status lanjut usia (0/1) |
| Partner | Status memiliki pasangan |
| Dependents | Status memiliki tanggungan |
| tenure | Lama berlangganan (bulan) |
| Contract | Jenis kontrak |
| MonthlyCharges | Biaya bulanan |
| TotalCharges | Total biaya |
| Churn | Target: status berhenti berlangganan |

*(Lengkapi tabel ini sesuai seluruh kolom dataset asli)*

### Exploratory Data Analysis (EDA)
- [ISI: insight dari visualisasi distribusi churn]
- [ISI: insight dari hubungan churn dengan tipe kontrak]
- [ISI: insight dari hubungan churn dengan tenure]

---

## 3. Data Preparation

Tahapan yang dilakukan:
1. **Konversi tipe data**: Kolom `TotalCharges` dikonversi dari string ke numerik, nilai kosong diisi dengan median.
2. **Penghapusan kolom**: Kolom `customerID` dihapus karena tidak relevan untuk pemodelan.
3. **Encoding**: Seluruh fitur kategorikal diubah menjadi numerik menggunakan Label Encoding.
4. **Splitting**: Data dibagi menjadi data latih (80%) dan data uji (20%) dengan stratifikasi pada target.
5. **Scaling**: Fitur numerik dinormalisasi menggunakan StandardScaler (khusus untuk model Logistic Regression).

**Alasan**: Setiap tahapan diperlukan agar data konsisten secara format, bebas dari nilai hilang, dan dalam skala yang sesuai untuk algoritma yang sensitif terhadap skala data.

---

## 4. Modeling

Tiga algoritma yang digunakan:

1. **Logistic Regression** — model linear sederhana sebagai baseline.
2. **Random Forest** — model ensemble berbasis decision tree, mampu menangkap hubungan non-linear.
3. **XGBoost** — model gradient boosting (poin plus, belum diajarkan di kelas), umumnya memberikan performa lebih baik pada data tabular.

### Parameter yang digunakan
- Logistic Regression: `max_iter=1000`, `random_state=42`
- Random Forest: `n_estimators=100`, `random_state=42`
- XGBoost: `random_state=42`, `eval_metric='logloss'`

### Model Terbaik
Berdasarkan hasil evaluasi pada tahap berikutnya, model **[ISI: nama model terbaik]** dipilih sebagai model final karena memiliki F1-Score tertinggi.

---

## 5. Evaluation

### Metrik yang digunakan
- **Accuracy**: proporsi prediksi yang benar dari seluruh data.
- **Precision**: proporsi prediksi churn yang benar-benar churn.
- **Recall**: proporsi pelanggan churn yang berhasil terdeteksi oleh model.
- **F1-Score**: rata-rata harmonik precision dan recall, cocok untuk data tidak seimbang.

### Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | [ISI] | [ISI] | [ISI] | [ISI] |
| Random Forest | [ISI] | [ISI] | [ISI] | [ISI] |
| XGBoost | [ISI] | [ISI] | [ISI] | [ISI] |

**Kesimpulan**: Model [ISI: nama model] dipilih sebagai model final karena [ISI: alasan berdasarkan tabel di atas].

---

## 6. Deployment

Model di-deploy menggunakan **Streamlit** sebagai aplikasi web interaktif yang memungkinkan pengguna memasukkan data pelanggan dan mendapatkan prediksi churn secara langsung.

### Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

### File Terkait
- `model_churn.pkl` — model machine learning yang telah dilatih
- `scaler.pkl` — objek scaler untuk normalisasi fitur
- `label_encoders.pkl` — encoder untuk fitur kategorikal
- `app.py` — aplikasi Streamlit untuk inference
- `requirements.txt` — daftar dependensi

---

## Tim
- [ISI: Nama Anggota 1] - [ISI: NIM]
- [ISI: Nama Anggota 2] - [ISI: NIM]
