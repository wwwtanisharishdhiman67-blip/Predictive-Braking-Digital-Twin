import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("🚀 Initiating AI Surrogate Model Training from Direct ANSYS Data...")

# ---------------------------------------------------------
# STEP 1: THE DATA (ANSYS Transient Thermal Results)
# ---------------------------------------------------------
time_s = np.array([0.1, 0.15229, 0.20458, 0.36146, 0.83209, 1.8321, 2.8321, 3.8321, 4.8321, 5.8321, 6.8321, 7.8321, 8.8321, 9.416, 10.0])
max_temp_c = np.array([23.265, 23.642, 23.933, 24.585, 26.176, 29.208, 35.892, 45.878, 58.781, 70.758, 82.175, 93.417, 104.56, 111.04, 117.5])

# Combining into a pandas DataFrame
data = pd.DataFrame({
    'Time': time_s,
    'Max_Temperature': max_temp_c
})

# ---------------------------------------------------------
# STEP 2: DEFINE INPUTS & OUTPUTS
# ---------------------------------------------------------
# Input (X): The AI will look at Time
X = data[['Time']] 

# Output (y): The AI will predict Maximum Temperature
y = data['Max_Temperature']

# ---------------------------------------------------------
# STEP 3: TRAIN-TEST SPLIT (80% Train, 20% Test)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# STEP 4: BUILD & TRAIN THE BRAIN (Random Forest)
# ---------------------------------------------------------
ai_model = RandomForestRegressor(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 5: TEST THE ACCURACY
# ---------------------------------------------------------
predictions = ai_model.predict(X_test)
accuracy = r2_score(y_test, predictions) * 100
mse = mean_squared_error(y_test, predictions)

print(f"✅ Training Complete!")
print(f"🔥 Model Accuracy (R2 Score): {accuracy:.2f}%")
print(f"📉 Mean Squared Error: {mse:.2f}")

# ---------------------------------------------------------
# STEP 6: EXPORT THE AI BRAIN
# ---------------------------------------------------------
joblib.dump(ai_model, 'brake_thermal_surrogate.pkl')
print("💾 Model saved successfully as 'brake_thermal_surrogate.pkl'. Ready for Phase 4 Dashboard!")