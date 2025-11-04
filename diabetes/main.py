from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np


# Define input structure
class InputData(BaseModel):
    bmi: float
    cholesterol_total: float
    smoking_status: float  # 0, 0.5, or 1
    alcohol_consumption_per_week: int
    physical_activity_minutes_per_week: float
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day:float
    family_history_diabetes: float
    hypertension_history: float
    cardiovascular_history: float
    waist_to_hip_ratio: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    cholesterol_total: float
    hdl_cholesterol:float
    ldl_cholesterol:float 
    triglycerides:float
    glucose_fasting: float
    glucose_postprandial: float
    insulin_level: float
    hba1c: float
  

# Load model pipeline (scaler + logistic regression)
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diabetes prediction API running"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
    data.smoking_status,
    data.alcohol_consumption_per_week,
    data.physical_activity_minutes_per_week,
    data.diet_score,
    data.sleep_hours_per_day,
    data.screen_time_hours_per_day,
    data.family_history_diabetes,
    data.hypertension_history,
    data.cardiovascular_history,
    data.bmi,
    data.waist_to_hip_ratio,
    data.systolic_bp,
    data.diastolic_bp,
    data.heart_rate,
    data.cholesterol_total,
    data.hdl_cholesterol,
    data.ldl_cholesterol,
    data.triglycerides,
    data.glucose_fasting,
    data.glucose_postprandial,
    data.insulin_level,
    data.hba1c
]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    print(prediction)
    return {"prediction": int(prediction[0])} 