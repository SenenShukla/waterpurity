import streamlit as st
import pickle
import numpy as np
import os

# Function to load the model and scaler safely
def load_assets():
    # Get the directory where the script is located to handle paths correctly
    base_path = os.path.dirname(__file__)
    
    scaler_path = os.path.join(base_path, 'scaler (1).pkl')
    model_path = os.path.join(base_path, 'best_model.pkl')

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"Error: Resource file not found. Ensure '{e.filename}' is in the repository.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# Load the assets
scaler, model = load_assets()

# --- Streamlit UI ---
st.set_page_config(page_title="Water Potability Prediction", page_icon="💧")

st.title("💧 Water Potability Prediction App")
st.markdown("""
Enter the water quality metrics below to predict if the water is safe for human consumption (**Potable**) or not.
""")

# Creating a grid for input fields
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
    chloramines = st.number_input("Chloramines (ppm)", value=7.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.0)

with col2:
    hardness = st.number_input("Hardness (mg/L)", value=196.0)
    sulfate = st.number_input("Sulfate (mg/L)", value=333.0)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.0)

with col3:
    solids = st.number_input("Solids (ppm)", value=20000.0)
    conductivity = st.number_input("Conductivity (μS/cm)", value=420.0)
    turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# Prediction Logic
if st.button("Predict Potability", type="primary"):
    # Create feature array in the exact order the scaler expects
    features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # 1. Scale the features
    scaled_features = scaler.transform(features)
    
    # 2. Predict
    prediction = model.predict(scaled_features)
    
    st.divider()
    
    if prediction[0] == 1:
        st.success("### Prediction: Potable (1)")
        st.write("✅ This water is safe for drinking.")
    else:
        st.error("### Prediction: Not Potable (0)")
        st.write("❌ This water is NOT safe for drinking.")

st.info("**Note:** This prediction is based on the Support Vector Machine (SVC) model provided.")