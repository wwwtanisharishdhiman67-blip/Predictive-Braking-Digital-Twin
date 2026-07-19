# 🚗 Predictive Braking Digital Twin 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/AI-Random_Forest-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Automotive_R%26D-orange)

> **Bridging the gap between Core Mechanical Engineering, Control Systems, and Artificial Intelligence to build a self-healing, predictive Vehicle ECU.**

## 📌 Project Overview
Traditional engineering relies on steady-state thermal analysis and static reports. This project takes a massive leap into **Industry 4.0** by developing a **Closed-Loop Predictive Digital Twin** for an Electric Vehicle's (EV) braking system. 

Instead of waiting 15 minutes for an FEA solver to calculate brake temperatures, this system uses a trained **Machine Learning Surrogate Model** to predict Transient Thermal behavior in **0.01 seconds**, allowing the Virtual ECU to intervene and prevent mechanical **Brake Fade** in real-time.

---

## ⚙️ System Architecture (The 4 Phases)

### 1. The Physics: Transient Thermal FEA (ANSYS Mechanical)
* Simulated a high-speed braking scenario (120 km/h to 0) on a full disc brake, caliper, and wheel hub assembly.
* Extracted 10 seconds of transient thermal dissipation data under extreme friction.

### 2. The Brain: Finite State Machine & ABS Logic (MATLAB / Simulink Stateflow)
* Modeled wheel-slip ratio and tire-road traction physics.
* Developed a Supervisory ECU using **Stateflow** with 3 distinct dynamic modes:
  * `NORMAL_BRAKING`: Standard hydraulic braking.
  * `ABS_RELEASE`: High-frequency Bang-Bang controller to maintain optimal slip ratio (preventing wheel lock).
  * `THERMAL_PROTECTION_MODE`: Active intervention state.

### 3. The AI Surrogate Model (Python / Scikit-Learn)
* Trained a **Random Forest Regressor** (`brake_thermal_surrogate.pkl`) on the ANSYS Transient data.
* The AI learns the complex multi-physics of heat generation and acts as an ultra-fast digital twin of the physical brake rotor.

### 4. The Cockpit: Live Telemetry Dashboard (Streamlit)
* Developed an interactive web-based dashboard simulating the EV's center console.
* **The Magic:** As the driver applies brakes, the AI predicts the temperature rise in real-time. If the prediction crosses the critical threshold (105°C), the system automatically triggers the `THERMAL_PROTECTION_MODE`, engaging regenerative braking to save the mechanical discs from failing.

---

## 📂 Repository Structure

| File Name | Description |
| :--- | :--- |
| `train_model.py` | The Python script used to train the Random Forest model on ANSYS data. |
| `brake_thermal_surrogate.pkl` | The exported AI Brain (Model weights). |
| `dashboard.py` | The Streamlit application script for the live Digital Twin UI. |
| `README.md` | Project documentation and architecture details. |

---

## 🚀 How to Run the Digital Twin Locally

Install the required dependencies:

Bash
pip install streamlit pandas numpy scikit-learn plotly joblib
Launch the Dashboard:

Bash
streamlit run dashboard.py
Conceptualized and Developed by Tanish Dhiman | R&D Simulation integrating ANSYS, MATLAB, and AI.*
   git clone [https://github.com/YOUR_USERNAME/Predictive-Braking-Digital-Twin.git](https://github.com/YOUR_USERNAME/Predictive-Braking-Digital-Twin.git)
   cd Predictive-Braking-Digital-Twin
