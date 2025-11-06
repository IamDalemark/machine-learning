from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from google import genai
import shap as sh
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
client = genai.Client()
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
    explainer = sh.LinearExplainer(model, scaled_features, feature_perturbation="interventional")
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names)
    response = client.models.generate_content(
    model="gemini-2.5-flash", contents=f"Explain how ${feature_names} works in a few words"
)
    return {"prediction": int(prediction[0]), "response":response.text}