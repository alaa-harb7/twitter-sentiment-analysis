from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model2.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ✅ التصنيفات الأربعة
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
