import streamlit as st
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource # Caches the model so it doesn't reload on every interaction
def load_assets():
    # Use absolute paths to prevent "File Not Found" errors on different environments
    base_path = os.path.dirname(__file__)
    scaler_path = os.path.join(base_path, 'scaler (1).pkl')
    model_path = os.path.join(base_path, 'best_model.pkl')

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

scaler, model = load_assets()

# --- UI DESIGN ---
st.title("🚰 Water Potability Prediction")
st.markdown("""
Enter the chemical properties of the water sample below to determine if it is safe for human consumption.
""")

if scaler and model:
    # Organize inputs into columns for a cleaner look
    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH Level (0-14)", min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.number_input("Hardness (mg/L)", value=196.3)
        solids = st.number_input("Solids (Total dissolved solids - ppm)", value=20000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.1)
        sulfate = st.number_input("Sulfate (mg/L)", value=333.7)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", value=426.2)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=14.1)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.3)
        turbidity = st.number_input("Turbidity (NTU)", value=3.9)

    st.markdown("---")

    # PREDICTION LOGIC
    if st.button("Analyze Water Quality", type="primary"):
        # Create input array (Must match the order of your scaler training)
        # Order: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        
        # 1. Scale the input
        scaled_data = scaler.transform(input_data)
        
        # 2. Predict
        prediction = model.predict(scaled_data)
        
        # 3. Output Results
        if prediction[0] == 1:
            st.success("### Result: Potable (1)")
            st.balloons()
            st.write("✅ **Safe for human consumption.** This water sample meets the safety standards according to the model.")
        else:
            st.error("### Result: Not Potable (0)")
            st.write("❌ **Unsafe for human consumption.** The chemical levels indicate this water is not potable.")

else:
    st.warning("Please ensure 'scaler (1).pkl' and 'best_model.pkl' are in the same folder as this script.")

# Information Section
with st.expander("About this App"):
    st.write("""
    This model uses a Support Vector Machine (SVM) to classify water safety. 
    Potability is defined as:
    - **1 (Potable):** Safe to drink.
    - **0 (Not Potable):** Unsafe to drink.
    """)