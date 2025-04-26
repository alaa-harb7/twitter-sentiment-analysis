from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import gdown  

# Model and Vectorizer URLs
model_url = "https://drive.google.com/uc?id=1lpGgmJgtFBCKxX9oaGEPC9lv3u1Y3r3Q"
vectorizer_url = "https://drive.google.com/uc?id=15TKBguU2r1ihDrjrPutwRBQGlXrpW6q1"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)
        print(f"{filename} downloaded.")

# Upload Model and Vectorizer From Drive
download_file(model_url, "sentiment_model2.pkl")
download_file(vectorizer_url, "vectorizer.pkl")

# Load model and vectorizer
model = joblib.load("sentiment_model2.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 4 Categories
label_map = {
    3: "Positive",
    1: "Negative",
    2: "Neutral",
    0: "Irrelevant"
}

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextInput):
    vector = vectorizer.transform([data.text])
    prediction = model.predict(vector)
    label = label_map.get(prediction[0], "Unknown")
    return {"prediction": label}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_from_form(request: Request, text: str = Form(...)):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    label = label_map.get(prediction[0], "Unknown")
    return templates.TemplateResponse("index.html", {"request": request, "result": label})
