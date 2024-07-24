import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load pre-trained models
rfr_model = joblib.load("model_rfr_bt.pkl")
gbm_model = joblib.load("model_gbm_bt.pkl")

# Load label encoders for categorical features
categorical_features = [ 'shift', 'jam', 'plant_model', 'material', 'alat_muat', 'no_hauler', 'disposal']
label_encoders = {column: joblib.load(f"le_{column}.pkl") for column in categorical_features}

# Load dataset to get unique values for selectboxes
dataset = pd.read_excel("produksi_tanah_liat.xlsx")

# Convert date column to string without time
dataset['tanggal'] = dataset['tanggal'].dt.date.astype(str)

# Function to get unique values for each feature
def get_unique_values(column_name):
    return dataset[column_name].unique()

st.title("Prediksi Volume Produksi Pertambangan Tanah Liat")

# Input features with unique values from dataset
# tanggal = st.selectbox("Masukkan tanggal", get_unique_values('tanggal'))
shift = st.selectbox("Pilih shift", get_unique_values('shift'))
jam = st.selectbox("Masukkan jam", get_unique_values('jam'))
plant_model = st.selectbox("Pilih plant model", get_unique_values('plant_model'))
material = st.selectbox("Pilih material", get_unique_values('material'))
kapasitas = st.number_input("Masukkan kapasitas", value=0.0)
ritase = st.number_input("Masukkan ritase", value=0.0)
alat_muat = st.selectbox("Pilih alat muat", get_unique_values('alat_muat'))
jarak = st.number_input("Masukkan jarak", value=0.0)
no_hauler = st.selectbox("Pilih nomor hauler", get_unique_values('no_hauler'))
disposal = st.selectbox("Pilih disposal", get_unique_values('disposal'))

# Encode the categorical inputs
input_data = pd.DataFrame({
    # 'tanggal': [tanggal],
    'shift': [shift],
    'jam': [jam],
    'plant_model': [plant_model],
    'material': [material],
    'kapasitas': [kapasitas],
    'ritase': [ritase],
    'alat_muat': [alat_muat],
    'jarak': [jarak],
    'no_hauler': [no_hauler],
    'disposal': [disposal]
})

# Ensure no empty columns in the DataFrame
input_data = input_data.loc[:, ~input_data.columns.str.contains('^Unnamed')]

# Function to encode data with error handling for unseen labels
def safe_transform(encoder, value):
    if value not in encoder.classes_:
        st.write(f"Value '{value}' not seen before for encoder. Adding to encoder.")
        # Add new label to the classes
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

# Preprocess categorical input data
for column in categorical_features:
    input_data[column] = input_data[column].apply(lambda x: safe_transform(label_encoders[column], x))

# Display encoded input data for debugging
# st.write("Input data encoded:", input_data)

def predict_rf(input_data):
    prediction = rfr_model.predict(input_data)
    return prediction

def predict_gbm(input_data):
    prediction = gbm_model.predict(input_data)
    return prediction

if st.button("Prediksi dengan Random Forest"):
    rf_prediction = predict_rf(input_data)
    st.write(f"Prediksi volume(ton) dengan Random Forest: {rf_prediction[0]}")

if st.button("Prediksi dengan Gradient Boosting Machine"):
    gbm_prediction = predict_gbm(input_data)
    st.write(f"Prediksi volume(ton) dengan Gradient Boosting Machine: {gbm_prediction[0]}")
