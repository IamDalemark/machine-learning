from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# model input
class InputData(BaseModel):
    research_document: str

# model pipeline

with open("thyroid_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# local api
app = FastAPI()

@app.get("/")
def root():
    return {"message", "thyroid cancer document predection working"}

@app.post("/predict")
def predict(data: InputData):
    features = vectorizer.transform([data.research_document])
    print(features)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    print(prediction[0])
    return {"prediction": int(prediction[0])}