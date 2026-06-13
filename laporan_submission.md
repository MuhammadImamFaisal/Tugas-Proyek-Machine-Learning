# Prediksi Customer Churn - Muhammad Imam Faisal & Muchamad Alfaidzin

## Domain Proyek

### Latar Belakang
Industri telekomunikasi merupakan sektor dengan tingkat persaingan yang sangat tinggi, di mana pelanggan memiliki banyak pilihan operator dan dapat dengan mudah berpindah ke layanan lain (churn). Kehilangan pelanggan (customer churn) berdampak langsung pada pendapatan perusahaan, sementara biaya untuk mendapatkan pelanggan baru jauh lebih besar dibandingkan biaya untuk mempertahankan pelanggan yang sudah ada. Oleh karena itu, kemampuan untuk memprediksi pelanggan mana yang berpotensi churn menjadi sangat penting agar perusahaan dapat melakukan tindakan retensi secara proaktif, seperti memberikan promo khusus atau peningkatan layanan kepada pelanggan yang berisiko tinggi.

### Masalah yang Harus Diselesaikan
Berdasarkan data historis pelanggan yang mencakup informasi demografi, layanan yang digunakan, jenis kontrak, dan riwayat tagihan, perusahaan perlu mengidentifikasi pola yang membedakan pelanggan yang akan churn dengan yang tidak. Masalah ini diselesaikan dengan membangun model klasifikasi yang dapat memprediksi probabilitas churn seorang pelanggan berdasarkan profil dan riwayat penggunaan layanannya.

Format Referensi: [Customer Churn Prediction in Telecom Industry](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Business Understanding

### Problem Statements
Bagaimana cara membangun model machine learning yang dapat memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) atau tidak, berdasarkan data demografi, layanan yang digunakan, jenis kontrak, dan informasi tagihan pelanggan?

### Goals
- Mengembangkan model klasifikasi yang dapat memprediksi status churn pelanggan dengan performa yang baik.
- Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap kemungkinan pelanggan melakukan churn, sehingga dapat digunakan sebagai dasar strategi retensi.

### Solution Statements
- Pendekatan pertama: Membangun model klasifikasi menggunakan algoritma **Logistic Regression** sebagai baseline, yang memodelkan hubungan linear antara fitur pelanggan dan probabilitas churn.
- Pendekatan kedua: Membangun model **Random Forest**, sebuah algoritma ensemble berbasis decision tree yang mampu menangkap hubungan non-linear antar fitur.
- Pendekatan ketiga (poin plus): Membangun model **XGBoost**, algoritma gradient boosting yang belum diajarkan di kelas, untuk dibandingkan performanya dengan dua model sebelumnya.

Evaluasi dari ketiga solusi ini menggunakan metrik **Accuracy, Precision, Recall, dan F1-Score**, dengan F1-Score sebagai metrik utama karena distribusi kelas target yang tidak seimbang.

## Data Understanding
Dataset yang digunakan adalah **Telco Customer Churn**, yang berisi data demografi, layanan, kontrak, dan tagihan pelanggan sebuah perusahaan telekomunikasi.

Sumber dataset: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Informasi Dataset**:
- Jumlah data: Dataset ini terdiri dari 7043 baris dan 21 kolom.
- Kondisi data:
    - Missing values: Tidak ditemukan missing value pada pengecekan awal menggunakan `isnull().sum()`. Namun, kolom `TotalCharges` memiliki nilai kosong berupa karakter spasi yang baru terdeteksi setelah dikonversi ke tipe numerik, dan ditangani dengan mengisi nilai median.
    - Duplikat: Tidak ditemukan data duplikat berdasarkan pengecekan `duplicated().sum()`.
    - Class imbalance: Distribusi target `Churn` tidak seimbang, dengan jumlah pelanggan yang tidak churn jauh lebih banyak dibandingkan yang churn.

**Fitur pada Telco Customer Churn Dataset adalah sebagai berikut**:
- customerID: ID unik pelanggan (dihapus saat preprocessing karena tidak relevan untuk model).
- gender: Jenis kelamin pelanggan.
- SeniorCitizen: Status lanjut usia pelanggan (0/1).
- Partner: Status memiliki pasangan.
- Dependents: Status memiliki tanggungan.
- tenure: Lama berlangganan dalam bulan.
- PhoneService: Status berlangganan layanan telepon.
- MultipleLines: Status memiliki lebih dari satu saluran telepon.
- InternetService: Jenis layanan internet (DSL/Fiber optic/No).
- OnlineSecurity: Status berlangganan layanan keamanan online.
- OnlineBackup: Status berlangganan layanan backup online.
- DeviceProtection: Status berlangganan layanan proteksi perangkat.
- TechSupport: Status berlangganan layanan dukungan teknis.
- StreamingTV: Status berlangganan layanan streaming TV.
- StreamingMovies: Status berlangganan layanan streaming film.
- Contract: Jenis kontrak (Month-to-month/One year/Two year).
- PaperlessBilling: Status menggunakan tagihan tanpa kertas.
- PaymentMethod: Metode pembayaran yang digunakan.
- MonthlyCharges: Biaya bulanan pelanggan.
- TotalCharges: Total biaya yang telah dibayarkan pelanggan.
- Churn: Target, status berhenti berlangganan (Yes/No).

### Exploratory Data Analysis (EDA)
Sebelum melanjutkan ke tahap modeling, dilakukan eksplorasi data untuk memahami distribusi target dan hubungan antar fitur menggunakan visualisasi countplot dan histogram.

**Visualisasi**:
- Distribusi Churn: Jumlah pelanggan yang tidak churn jauh lebih banyak dibandingkan yang churn, menunjukkan adanya class imbalance pada target.
- Churn berdasarkan Tipe Kontrak: Pelanggan dengan kontrak month-to-month memiliki proporsi churn yang jauh lebih tinggi dibandingkan pelanggan dengan kontrak one year maupun two year.
- Distribusi Tenure vs Churn: Pelanggan dengan tenure rendah (pelanggan baru) lebih cenderung melakukan churn dibandingkan pelanggan dengan tenure tinggi (pelanggan lama).

**Insight**: Pelanggan dengan kontrak month-to-month dan tenure rendah lebih cenderung churn, sehingga kedua fitur ini menjadi indikator penting dalam memprediksi churn.

## Data Preparation
Proses data preparation yang dilakukan pada dataset ini mencakup langkah-langkah berikut:
- **Konversi Tipe Data**: Kolom `TotalCharges` dikonversi dari tipe string ke numerik menggunakan `pd.to_numeric()` dengan parameter `errors='coerce'`, kemudian nilai yang kosong (NaN) diisi dengan median dari kolom tersebut.
- **Penghapusan Kolom**: Kolom `customerID` dihapus karena bersifat unik untuk setiap baris dan tidak memberikan informasi yang berguna bagi model.
- **Encoding Target**: Kolom `Churn` yang awalnya berisi nilai kategorikal (`Yes`/`No`) diubah menjadi numerik (1/0) menggunakan mapping.
- **Encoding Fitur Kategorikal**: Seluruh fitur bertipe `object` diubah menjadi numerik menggunakan `LabelEncoder` dari `sklearn.preprocessing`, dengan setiap encoder disimpan dalam dictionary `le_dict` agar dapat digunakan kembali pada tahap deployment.
- **Pembagian Data**: Dataset dibagi menjadi data latih (5634 baris) dan data uji (1409 baris) dengan komposisi 80:20 menggunakan `train_test_split`, dengan parameter `stratify=y` untuk menjaga proporsi kelas target tetap seimbang di kedua subset.
- **Normalisasi**: Fitur-fitur pada data latih dan uji dinormalisasi menggunakan `StandardScaler` dari `sklearn.preprocessing`, khusus digunakan untuk model Logistic Regression yang sensitif terhadap perbedaan skala antar fitur.

**Alasan Data Preparation**:
Langkah-langkah ini penting untuk memastikan data siap digunakan dalam model machine learning. Konversi tipe data dan penanganan nilai kosong memastikan data konsisten secara format, encoding mengubah data kategorikal menjadi bentuk numerik yang dapat diproses oleh algoritma, pembagian data dengan stratifikasi memastikan model dapat dievaluasi secara adil pada data yang belum pernah dilihat sebelumnya, dan normalisasi mencegah bias akibat perbedaan skala antar fitur pada model yang sensitif terhadap skala seperti Logistic Regression.

## Model Development
Pada bagian ini, dikembangkan tiga model klasifikasi untuk memprediksi status churn pelanggan, yaitu Logistic Regression, Random Forest, dan XGBoost.

### 1. Logistic Regression
Logistic Regression memodelkan probabilitas churn sebagai fungsi linear dari fitur input yang dilewatkan melalui fungsi sigmoid, sehingga menghasilkan nilai probabilitas antara 0 dan 1. Model ini dilatih menggunakan data yang telah dinormalisasi (`X_train_scaled`).

**Parameter yang Digunakan**:
- max_iter: 1000 - Jumlah maksimum iterasi yang dilakukan solver untuk mencapai konvergensi.
- random_state: 42 - Memastikan hasil yang dapat direproduksi.

### 2. Random Forest
Random Forest adalah algoritma ensemble yang membangun banyak decision tree dari subset data dan fitur yang dipilih secara acak, kemudian menggabungkan hasil prediksi dari seluruh tree melalui voting mayoritas. Model ini dilatih menggunakan data tanpa normalisasi (`X_train`).

**Parameter yang Digunakan**:
- n_estimators: 100 - Jumlah decision tree yang dibangun dalam ensemble.
- random_state: 42 - Memastikan hasil yang dapat direproduksi.

### 3. XGBoost (Poin Plus)
XGBoost adalah algoritma gradient boosting yang membangun decision tree secara bertahap, di mana setiap tree baru dibangun untuk memperbaiki kesalahan (residual) dari tree-tree sebelumnya. Algoritma ini belum diajarkan di kelas dan diimplementasikan sebagai bagian dari eksplorasi mandiri. Model ini dilatih menggunakan data tanpa normalisasi (`X_train`).

**Parameter yang Digunakan**:
- eval_metric: 'logloss' - Metrik evaluasi internal yang digunakan selama proses training.
- random_state: 42 - Memastikan hasil yang dapat direproduksi.

### Pemilihan Model Terbaik
Berdasarkan hasil evaluasi pada bagian berikutnya, model **Logistic Regression** dipilih sebagai model terbaik karena memiliki F1-Score tertinggi di antara ketiga model yang diuji.

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah Accuracy, Precision, Recall, dan F1-Score, yang umum digunakan untuk mengevaluasi performa model klasifikasi.

**Penjelasan Metrik**:
- **Accuracy** mengukur proporsi prediksi yang benar dari seluruh data, dihitung dengan rumus (TP+TN)/(TP+TN+FP+FN).
- **Precision** mengukur proporsi prediksi positif (churn) yang benar-benar positif, dihitung dengan rumus TP/(TP+FP).
- **Recall** mengukur proporsi data positif (churn) yang berhasil terdeteksi oleh model, dihitung dengan rumus TP/(TP+FN).
- **F1-Score** merupakan rata-rata harmonik dari Precision dan Recall, dihitung dengan rumus 2 x (Precision x Recall)/(Precision + Recall), dan digunakan sebagai metrik utama karena lebih representatif untuk data dengan distribusi kelas yang tidak seimbang.

**Hasil Evaluasi**:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.7991 | 0.6426 | 0.5481 | **0.5916** |
| Random Forest | 0.7921 | 0.6373 | 0.5027 | 0.5620 |
| XGBoost | 0.7786 | 0.5957 | 0.5160 | 0.5530 |

- Logistic Regression memperoleh F1-Score tertinggi sebesar 0.5916, dengan jumlah False Negative (pelanggan churn yang tidak terdeteksi) sebanyak 169, lebih rendah dibandingkan Random Forest (186) dan XGBoost (181).
- Random Forest memperoleh F1-Score sebesar 0.5620, dengan Accuracy tertinggi di antara ketiga model namun Recall yang lebih rendah dibandingkan Logistic Regression.
- XGBoost memperoleh F1-Score sebesar 0.5530, performa paling rendah di antara ketiga model pada eksperimen ini.

**Dampak terhadap Business Understanding**:
- **Problem Statement**: Model Logistic Regression berhasil menjawab problem statement dengan memprediksi status churn pelanggan berdasarkan fitur demografi, layanan, kontrak, dan tagihan, dengan F1-Score sebesar 0.5916.
- **Goals**: Model yang dikembangkan mampu mengidentifikasi pelanggan berisiko churn, dan berdasarkan eksplorasi data, fitur jenis kontrak (Contract) dan lama berlangganan (tenure) teridentifikasi sebagai faktor yang berpengaruh signifikan terhadap churn, sehingga dapat menjadi dasar strategi retensi perusahaan.
- **Solution Statement**: Dari ketiga algoritma yang dibandingkan, Logistic Regression memberikan hasil F1-Score terbaik, diikuti oleh Random Forest dan XGBoost. Hasil ini menunjukkan bahwa untuk dataset ini, model linear sederhana sudah cukup kompetitif dibandingkan model ensemble yang lebih kompleks.

## Deployment
Model terbaik (Logistic Regression) di-deploy menggunakan **Streamlit** sebagai aplikasi web interaktif, yang memungkinkan pengguna memasukkan data pelanggan secara manual melalui form input dan memperoleh hasil prediksi churn beserta probabilitasnya secara langsung.

### Cara Menjalankan Aplikasi
1. Pastikan seluruh file (`app.py`, `model_churn.pkl`, `scaler.pkl`, `label_encoders.pkl`, `requirements.txt`) berada dalam satu folder.
2. Buka CMD/Terminal, arahkan ke folder tersebut menggunakan perintah `cd`, contoh:
   ```bash
   cd D:\proyek_churn
   ```
3. Install seluruh library yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

### Tampilan Aplikasi
Berikut adalah tampilan aplikasi Streamlit yang telah berjalan, berisi form input data pelanggan untuk memprediksi status churn:

![Tampilan Aplikasi Streamlit](./streamlit_app.png)

## Kesimpulan
Proyek ini menunjukkan bahwa model Logistic Regression dapat digunakan untuk memprediksi customer churn pada industri telekomunikasi dengan F1-Score sebesar 0.5916, mengungguli Random Forest dan XGBoost pada eksperimen ini. Fitur jenis kontrak dan lama berlangganan teridentifikasi sebagai faktor penting yang memengaruhi churn, sehingga dapat digunakan oleh perusahaan sebagai dasar dalam merancang strategi retensi pelanggan yang lebih tepat sasaran.
