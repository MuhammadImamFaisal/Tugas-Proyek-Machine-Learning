import streamlit as st
import pandas as pd
import joblib

# Load model & preprocessing tools
model = joblib.load('model_churn.pkl')       # Logistic Regression (model terbaik)
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

st.title("Prediksi Customer Churn")
st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")

# ===== Input semua fitur sesuai urutan kolom X_train =====
gender = st.selectbox("Gender", le_dict['gender'].classes_)
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", le_dict['Partner'].classes_)
dependents = st.selectbox("Dependents", le_dict['Dependents'].classes_)
tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", le_dict['PhoneService'].classes_)
multiple_lines = st.selectbox("Multiple Lines", le_dict['MultipleLines'].classes_)
internet_service = st.selectbox("Internet Service", le_dict['InternetService'].classes_)
online_security = st.selectbox("Online Security", le_dict['OnlineSecurity'].classes_)
online_backup = st.selectbox("Online Backup", le_dict['OnlineBackup'].classes_)
device_protection = st.selectbox("Device Protection", le_dict['DeviceProtection'].classes_)
tech_support = st.selectbox("Tech Support", le_dict['TechSupport'].classes_)
streaming_tv = st.selectbox("Streaming TV", le_dict['StreamingTV'].classes_)
streaming_movies = st.selectbox("Streaming Movies", le_dict['StreamingMovies'].classes_)
contract = st.selectbox("Contract", le_dict['Contract'].classes_)
paperless_billing = st.selectbox("Paperless Billing", le_dict['PaperlessBilling'].classes_)
payment_method = st.selectbox("Payment Method", le_dict['PaymentMethod'].classes_)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

if st.button("Prediksi"):
    # Susun input sesuai urutan kolom X_train (HARUS SAMA URUTANNYA)
    input_dict = {
        'gender': le_dict['gender'].transform([gender])[0],
        'SeniorCitizen': senior,
        'Partner': le_dict['Partner'].transform([partner])[0],
        'Dependents': le_dict['Dependents'].transform([dependents])[0],
        'tenure': tenure,
        'PhoneService': le_dict['PhoneService'].transform([phone_service])[0],
        'MultipleLines': le_dict['MultipleLines'].transform([multiple_lines])[0],
        'InternetService': le_dict['InternetService'].transform([internet_service])[0],
        'OnlineSecurity': le_dict['OnlineSecurity'].transform([online_security])[0],
        'OnlineBackup': le_dict['OnlineBackup'].transform([online_backup])[0],
        'DeviceProtection': le_dict['DeviceProtection'].transform([device_protection])[0],
        'TechSupport': le_dict['TechSupport'].transform([tech_support])[0],
        'StreamingTV': le_dict['StreamingTV'].transform([streaming_tv])[0],
        'StreamingMovies': le_dict['StreamingMovies'].transform([streaming_movies])[0],
        'Contract': le_dict['Contract'].transform([contract])[0],
        'PaperlessBilling': le_dict['PaperlessBilling'].transform([paperless_billing])[0],
        'PaymentMethod': le_dict['PaymentMethod'].transform([payment_method])[0],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Model terbaik = Logistic Regression -> WAJIB di-scale
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    result = "Churn" if pred[0] == 1 else "Tidak Churn"
    st.success(f"Hasil Prediksi: **{result}**")
    st.write(f"Probabilitas churn: {proba:.2%}")
