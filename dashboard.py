import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. LOAD THE AI BRAIN
# ---------------------------------------------------------
# Ye wahi file hai jo tune abhi pichle step mein banayi thi
model = joblib.load('brake_thermal_surrogate.pkl')

# ---------------------------------------------------------
# 2. DASHBOARD UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="EV Digital Twin", layout="wide", initial_sidebar_state="expanded")
st.title("🚗 AI-Powered Predictive Digital Twin")
st.markdown("### Dynamic Thermal Management & Brake Fade Control")
st.markdown("---")

# ---------------------------------------------------------
# 3. THE DRIVER'S INPUT (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Live Telemetry Input")
st.sidebar.markdown("Simulate braking duration to see AI prediction in real-time.")

# Slider to simulate how long the brakes are applied
time_input = st.sidebar.slider("Braking Duration (Seconds)", min_value=0.0, max_value=15.0, value=1.0, step=0.1)

# ---------------------------------------------------------
# 4. AI PREDICTION LOGIC
# ---------------------------------------------------------
# Wrap input in dataframe to match training format
input_data = pd.DataFrame({'Time': [time_input]})
predicted_temp = model.predict(input_data)[0]

# ---------------------------------------------------------
# 5. VISUALIZING THE DATA (GAUGES & WARNINGS)
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Live Brake Rotor Temperature")
    
    # Plotly Gauge Chart (Looks extremely professional)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_temp,
        number = {'suffix': " °C", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [20, 130], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [20, 80], 'color': "#00cc96"},   # Safe (Green)
                {'range': [80, 105], 'color': "#ffa15a"},  # Warning (Orange)
                {'range': [105, 130], 'color': "#ef553b"}  # Danger (Red)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 110
            }
        }
    ))
    # Make it fit the column nicely
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🧠 Adaptive ECU Status")
    st.markdown("<br>", unsafe_allow_html=True) # Adds some spacing
    
    # The Core Logic: AI reacting to the temperature
    if predicted_temp >= 105:
        st.error("🚨 **CRITICAL HEAT DETECTED: BRAKE FADE IMMINENT**")
        st.warning("⚡ **AI INTERVENTION ACTIVE:** Engaging Regenerative Braking. Limiting hydraulic brake pressure to prevent failure.")
    elif predicted_temp >= 80:
        st.warning("⚠️ **ELEVATED TEMPERATURE**")
        st.info("System monitoring closely. Pre-charging regen systems.")
    else:
        st.success("✅ **SYSTEM OPTIMAL**")
        st.write("Brake temperatures are within normal operating margins.")

st.markdown("---")
st.caption("Developed by Tanish Dhiman | R&D Simulation integrating ANSYS Transient FEA & Random Forest ML")