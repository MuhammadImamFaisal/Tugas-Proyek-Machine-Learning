import streamlit as st
import pandas as pd
import joblib

# Load model & preprocessing tools
model = joblib.load('model_churn.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

st.title("Prediksi Customer Churn")
st.write("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")

# Contoh input (sesuaikan dengan kolom dataset asli, urutan harus sama)
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

if st.button("Prediksi"):
    # Susun input sesuai urutan kolom training (CONTOH - sesuaikan!)
    input_dict = {
        'gender': le_dict['gender'].transform([gender])[0],
        'SeniorCitizen': senior,
        'Partner': le_dict['Partner'].transform([partner])[0],
        'Dependents': le_dict['Dependents'].transform([dependents])[0],
        'tenure': tenure,
        'Contract': le_dict['Contract'].transform([contract])[0],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # TODO: tambahkan kolom lain sesuai dataset (PhoneService, InternetService, dll)
        # gunakan le_dict[<nama_kolom>].transform([nilai])[0] untuk kategorikal
    }

    input_df = pd.DataFrame([input_dict])

    # Jika model butuh scaling (Logistic Regression), uncomment baris berikut:
    # input_scaled = scaler.transform(input_df)
    # pred = model.predict(input_scaled)

    pred = model.predict(input_df)
    result = "Churn" if pred[0] == 1 else "Tidak Churn"
    st.success(f"Hasil Prediksi: **{result}**")
