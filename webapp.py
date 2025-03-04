import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("Iris Flower Classifier 🌸")

st.write("Enter petal and sepal dimensions to predict the flower species.")

# User input
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Predict Button
if st.button("Predict"):
    # Prepare input
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.success(f"The predicted species is: **{prediction}**")
