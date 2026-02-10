import streamlit as st
import numpy as np
import joblib

model = joblib.load("LightGB_Customer_Segmentation.joblib")
scaler = joblib.load("rfm_scaler.joblib")

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("Customer Segmentation")
st.write("Predict customer segment using RFM values")

recency = st.number_input("Recency (days)", min_value=0) # Days since last purchase
frequency = st.number_input("Frequency", min_value=0) # Number of purchase
monetary = st.number_input("Monetary Value", min_value=0.0) # Total Spend

if st.button("Predict Customer Segment"):
    X_data = np.array([[recency, frequency, monetary]])
    X_log = np.log1p(X_data)
    X_scaled = scaler.transform(X_log)
    pred = model.predict(X_scaled)[0]

    
    segment_map = {
    "0-Regular customers": {
        "name": "Regular Customer",
        "color": "green"
    },
    "1-Loyal customers": {
        "name": "Loyal Customer",
        "color": "green"
    },
    "2-Churned customers": {
        "name": "Churned Customer",
        "color": "yellow"
    },
    "3-At risk customers": {
        "name": "At Risk Customer",
        "color": "red"
    }
    }

    segment_name = segment_map[pred]["name"]
    segment_color = segment_map[pred]["color"] 

    st.markdown(f"Predicted Segment: :{segment_color}[{segment_name}]")
