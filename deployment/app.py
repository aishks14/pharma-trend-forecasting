# Rossmann Store Sales Prediction Web App
# Streamlit Deployment

import streamlit as st
import pandas as pd
import joblib
import os
import logging
import warnings

# Silence sklearn/lightgbm warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Logger Configuration
# ---------------------------

LOG_FILE = "app.log"

logging.basicConfig(
filename=LOG_FILE,
filemode="a",
level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("Application started")

# ---------------------------
# Load Model Pipeline
# ---------------------------

MODEL_PATH = os.path.join("..", "models", "sales_pipeline_latest.pkl")

@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Model pipeline loaded successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error("Model could not be loaded. Please check the model file.")
        return None
    
pipeline = load_pipeline()

# ---------------------------
# Streamlit Page Settings
# ---------------------------

st.set_page_config(
page_title="Rossmann Store Sales Predictor",
page_icon="📊",
layout="centered"
)

st.title("📊 Rossmann Store Sales Prediction")
st.write("Predict daily sales for a Rossmann store based on store conditions.")

# ---------------------------
# User Inputs
# ---------------------------

st.header("Enter Store Details")

storetype = st.selectbox("Store Type", ["a", "b", "c", "d"])
assortment = st.selectbox("Assortment Type", ["a", "b", "c"])
stateholiday = st.selectbox("State Holiday", ["0", "a", "b", "c"])

customers = st.number_input("Number of Customers", min_value=0, value=650)
competitiondistance = st.number_input("Competition Distance (meters)", min_value=0, value=450)
promo = st.selectbox("Is Promotion Active?", [0, 1])

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Sales"):
    if pipeline is None:
        st.error("Model is not available.")
    else:
        try:
            # Create dataframe with EXACT feature schema
            input_df = pd.DataFrame({
                "storetype": [storetype],
                "assortment": [assortment],
                "stateholiday": [stateholiday],
                "customers": [customers],
                "competitiondistance": [competitiondistance],
                "promo": [promo]
            })

            logger.info(f"User input received: {input_df.to_dict()}")

            # Predict
            prediction = pipeline.predict(input_df)
            predicted_sales = int(prediction[0])

            logger.info(f"Prediction generated: {predicted_sales}")

            # Display result
            st.success(f"💰 Predicted Store Sales: ₹ {predicted_sales:,}")

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            st.error("Prediction failed. Please check input values.")

# ---------------------------
# Footer
# ---------------------------

st.markdown("---")
st.caption("Machine Learning Project — Rossmann Sales Forecasting")
