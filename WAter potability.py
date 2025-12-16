import streamlit as st
import pickle
import numpy as np

# Load the scaler and the model
def load_assets():
    with open('scaler (1).pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_assets()

# App Title and Description
st.title("🚰 Water Potability Predictor")
st.write("""
This app predicts if a water sample is **Potable (Safe to Drink)** or **Not Potable** based on its chemical properties.
""")

# Define input features based on the scaler's requirements 
# Features in order: ph, Hardness, Solids, Chloramines, Sulfate, 
# Conductivity, Organic_carbon, Trihalomethanes, Turbidity
st.sidebar.header("Water Quality Metrics")

def user_input_features():
    ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider("Hardness (mg/L)", 47.0, 323.0, 196.0)
    solids = st.sidebar.slider("Solids (ppm)", 320.0, 61227.0, 20927.0)
    chloramines = st.sidebar.slider("Chloramines (ppm)", 0.3, 13.0, 7.1)
    sulfate = st.sidebar.slider("Sulfate (mg/L)", 129.0, 481.0, 333.0)
    conductivity = st.sidebar.slider("Conductivity (μS/cm)", 181.0, 753.0, 421.0)
    organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 2.0, 28.0, 14.0)
    trihalomethanes = st.sidebar.slider("Trihalomethanes (μg/L)", 0.7, 124.0, 66.0)
    turbidity = st.sidebar.slider("Turbidity (NTU)", 1.4, 6.7, 3.9)
    
    data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                      conductivity, organic_carbon, trihalomethanes, turbidity]])
    return data

# Get user input
input_data = user_input_features()

# Predict Button
if st.button("Predict Potability"):
    # 1. Scale the input data using the loaded scaler 
    scaled_data = scaler.transform(input_data)
    
    # 2. Make prediction using the SVM model 
    prediction = model.predict(scaled_data)
    
    # Display Results
    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("✅ The water is **POTABLE** (Safe for consumption).")
    else:
        st.error("❌ The water is **NOT POTABLE** (Unsafe for consumption).")

# Optional: Display the feature values used
st.write("### Current Input Parameters")
st.write(input_data)