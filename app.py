
import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and label encoder
model = joblib.load('random_forest_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit app title
st.title('Dynamic Supply Chain Risk Classification')
st.write('Enter the feature values to predict the risk classification.')

# Feature names (excluding 'timestamp', 'risk_classification', 'delivery_time_deviation')
feature_names = [
    'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
    'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
    'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status',
    'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
    'supplier_reliability_score', 'lead_time_days', 'historical_demand',
    'iot_temperature', 'cargo_condition_status', 'route_risk_level',
    'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
    'disruption_likelihood_score', 'delay_probability'
]

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(f'Enter {feature}', value=0.0)

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Scale the input features
input_scaled = scaler.transform(input_df)

if st.button('Predict Risk'):
    # Make prediction
    prediction_encoded = model.predict(input_scaled)

    # Decode the prediction
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.success(f'The predicted risk classification is: **{prediction_label[0]}**')

