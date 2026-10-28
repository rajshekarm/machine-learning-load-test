import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model once when the application starts
artifact = joblib.load("model.joblib")

model = artifact["model"]
class_names = artifact["class_names"]

app = FastAPI(title="Iris Inference API")


# Define the input expected by the model
class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    return {
        "predicted_class": class_names[prediction],
        "probabilities": {
            name: round(float(probability), 4)
            for name, probability in zip(
                class_names, probabilities
            )
        },
    }