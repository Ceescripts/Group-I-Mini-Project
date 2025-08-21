import streamlit as st
import pickle
import pandas as pd

# Load preprocessor and model
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💻 Laptop Price Predictor")

# Numeric inputs
inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6)
weight_kg = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)

# Categorical inputs
company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "Other"])
typename = st.selectbox("Type", ["Notebook", "Ultrabook", "Gaming", "Workstation", "Other"])
opsys = st.selectbox("Operating System", ["Windows", "MacOS", "Linux", "No OS", "Other"])
cpu = st.text_input("CPU", "Intel Core i5")
gpu = st.text_input("GPU", "Nvidia GTX 1650")
laptop_id = st.text_input("Laptop ID", "12345")
memory = st.text_input("Memory", "512GB SSD")
product = st.text_input("Product Name", "Inspiron 15")
screenresolution = st.text_input("Screen Resolution", "1920x1080")

# Make prediction
if st.button("Predict Price"):
    # Build DataFrame with same columns as training
    input_df = pd.DataFrame({
        "inches": [inches],
        "weight_kg": [weight_kg],
        "ram_gb": [ram_gb],
        "company": [company],
        "typename": [typename],
        "opsys": [opsys],
        "cpu": [cpu],
        "gpu": [gpu],
        "laptop_id": [laptop_id],
        "memory": [memory],
        "product": [product],
        "screenresolution": [screenresolution],
    })

    # Preprocess + predict
    input_prep = preprocessor.transform(input_df)
    prediction = model.predict(input_prep)

    st.success(f"Estimated Price: €{prediction[0]:,.2f}")
