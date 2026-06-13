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

Dataset yang digunakan adalah **Telco Customer Churn** dari Kaggle, terdiri dari **7043 baris** dan **21 kolom** (20 fitur + 1 target).

### Fitur pada Dataset
| Fitur | Deskripsi |
|---|---|
| customerID | ID unik pelanggan (dihapus saat preprocessing) |
| gender | Jenis kelamin pelanggan |
| SeniorCitizen | Status lanjut usia (0/1) |
| Partner | Status memiliki pasangan |
| Dependents | Status memiliki tanggungan |
| tenure | Lama berlangganan (bulan) |
| PhoneService | Status berlangganan layanan telepon |
| MultipleLines | Status memiliki lebih dari satu saluran telepon |
| InternetService | Jenis layanan internet (DSL/Fiber optic/No) |
| OnlineSecurity | Status berlangganan layanan keamanan online |
| OnlineBackup | Status berlangganan layanan backup online |
| DeviceProtection | Status berlangganan layanan proteksi perangkat |
| TechSupport | Status berlangganan layanan dukungan teknis |
| StreamingTV | Status berlangganan layanan streaming TV |
| StreamingMovies | Status berlangganan layanan streaming film |
| Contract | Jenis kontrak (Month-to-month/One year/Two year) |
| PaperlessBilling | Status menggunakan tagihan tanpa kertas |
| PaymentMethod | Metode pembayaran |
| MonthlyCharges | Biaya bulanan |
| TotalCharges | Total biaya |
| Churn | Target: status berhenti berlangganan (Yes/No) |

### Kondisi Data
- Tidak ditemukan data duplikat.
- Tidak ada missing value pada data awal, namun kolom `TotalCharges` memiliki beberapa nilai kosong (berbentuk spasi) yang baru terdeteksi setelah dikonversi ke tipe numerik.
- Distribusi target tidak seimbang: mayoritas pelanggan tidak churn dibanding yang churn.

### Exploratory Data Analysis (EDA)
- Jumlah pelanggan yang tidak churn jauh lebih banyak dibanding yang churn, menunjukkan **ketidakseimbangan kelas (class imbalance)**.
- Pelanggan dengan kontrak **month-to-month** memiliki proporsi churn yang jauh lebih tinggi dibanding pelanggan dengan kontrak one year atau two year.
- Pelanggan dengan **tenure rendah** (pelanggan baru) memiliki kecenderungan churn yang lebih tinggi dibanding pelanggan dengan tenure tinggi (pelanggan lama).

**Insight**: Pelanggan dengan kontrak month-to-month dan tenure rendah lebih cenderung churn.

---

## 3. Data Preparation

Tahapan yang dilakukan secara berurutan:

1. **Konversi tipe data**: Kolom `TotalCharges` dikonversi dari string ke numerik menggunakan `pd.to_numeric` dengan `errors='coerce'`, nilai yang gagal dikonversi (kosong) diisi dengan median dari kolom tersebut.
2. **Penghapusan kolom**: Kolom `customerID` dihapus karena bersifat unik per baris dan tidak relevan untuk pemodelan.
3. **Encoding target**: Kolom `Churn` diubah dari kategori (`Yes`/`No`) menjadi numerik (1/0).
4. **Encoding fitur kategorikal**: Seluruh fitur bertipe `object` diubah menjadi numerik menggunakan `LabelEncoder`, dengan encoder masing-masing kolom disimpan dalam `le_dict` agar dapat digunakan kembali saat inference.
5. **Splitting data**: Data dibagi menjadi data latih (5634 baris) dan data uji (1409 baris), dengan rasio 80:20 dan stratifikasi pada target agar proporsi kelas tetap seimbang di kedua subset.
6. **Scaling**: Fitur pada data latih dan uji dinormalisasi menggunakan `StandardScaler`, khusus digunakan untuk model Logistic Regression yang sensitif terhadap skala fitur.

**Alasan**: Setiap tahapan diperlukan agar data konsisten secara format, bebas dari nilai hilang, dalam bentuk numerik yang dapat diproses model, dan dalam skala yang sesuai untuk algoritma berbasis jarak/linear seperti Logistic Regression.

---

## 4. Modeling

Tiga algoritma yang digunakan:

1. **Logistic Regression** — model linear sederhana sebagai baseline, dilatih menggunakan data yang telah dinormalisasi (`X_train_scaled`).
2. **Random Forest** — model ensemble berbasis decision tree, mampu menangkap hubungan non-linear antar fitur, dilatih menggunakan data tanpa scaling.
3. **XGBoost** — model gradient boosting (**poin plus**, belum diajarkan di kelas), dilatih menggunakan data tanpa scaling.

### Parameter yang digunakan
- Logistic Regression: `max_iter=1000`, `random_state=42`
- Random Forest: `n_estimators=100`, `random_state=42`
- XGBoost: `random_state=42`, `eval_metric='logloss'`

### Cara Kerja Singkat
- **Logistic Regression** memodelkan probabilitas churn sebagai fungsi linear dari fitur input yang dilewatkan fungsi sigmoid.
- **Random Forest** membangun banyak decision tree dari subset data dan fitur secara acak, kemudian menggabungkan prediksinya melalui voting.
- **XGBoost** membangun decision tree secara bertahap (boosting), di mana setiap tree baru berfokus memperbaiki kesalahan dari tree sebelumnya.

---

## 5. Evaluation

### Metrik yang digunakan
- **Accuracy**: proporsi prediksi yang benar dari seluruh data.
- **Precision**: dari semua pelanggan yang diprediksi churn, berapa proporsi yang benar-benar churn.
- **Recall**: dari semua pelanggan yang benar-benar churn, berapa proporsi yang berhasil terdeteksi oleh model.
- **F1-Score**: rata-rata harmonik precision dan recall, digunakan sebagai metrik utama karena data target tidak seimbang.

### Hasil Evaluasi

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.7991 | 0.6426 | 0.5481 | **0.5916** |
| Random Forest | 0.7921 | 0.6373 | 0.5027 | 0.5620 |
| XGBoost | 0.7786 | 0.5957 | 0.5160 | 0.5530 |

### Model Terbaik
Model **Logistic Regression** dipilih sebagai model final karena memiliki **F1-Score tertinggi (0.5916)** di antara ketiga model, meskipun selisihnya tidak terlalu besar dengan Random Forest. F1-Score dipilih sebagai acuan utama karena lebih representatif untuk kasus data dengan distribusi kelas tidak seimbang dibandingkan Accuracy semata. Confusion matrix Logistic Regression juga menunjukkan jumlah False Negative (169) yang lebih rendah dibanding kedua model lain, yang penting dalam konteks churn karena False Negative berarti pelanggan yang sebenarnya akan churn tidak terdeteksi oleh model.

---

## 6. Deployment

Model di-deploy menggunakan **Streamlit** sebagai aplikasi web interaktif yang memungkinkan pengguna memasukkan data pelanggan dan mendapatkan prediksi churn beserta probabilitasnya secara langsung.

### Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

### File Terkait
- `model_churn.pkl` — model Logistic Regression (model terbaik) yang telah dilatih
- `scaler.pkl` — objek StandardScaler untuk normalisasi fitur, digunakan saat inference karena model terbaik adalah Logistic Regression
- `label_encoders.pkl` — dictionary LabelEncoder untuk setiap fitur kategorikal
- `app.py` — aplikasi Streamlit untuk inference
- `requirements.txt` — daftar dependensi
- `Proyek_Churn.ipynb` — notebook lengkap proses CRISP-DM dari Business Understanding hingga Deployment

---

## Tim
- [ISI: Nama Anggota 1] - [ISI: NIM]
- [ISI: Nama Anggota 2] - [ISI: NIM]
