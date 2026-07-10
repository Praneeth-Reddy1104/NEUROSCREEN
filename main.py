"""
main.py
-------
FastAPI backend for the NeuroScreen ASD screening app.
Serves the NeuroScreen HTML frontend at "/" and exposes a JSON
"/predict" endpoint that the page's own JS calls via fetch().

Run with:
    uvicorn main:app --reload --port 8000
Then open:
    http://localhost:8000/
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from model_utils import ModelNotTrainedError, asd_model
from schemas import ASDInput, ASDPrediction

app = FastAPI(
    title="NeuroScreen API",
    description="ASD screening prediction API (AQ-10 based).",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

# allow the page's own fetch() calls (and any other local frontend) to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    asd_model.load()
    if asd_model.loaded:
        print("✅ Model artifacts loaded.")
    else:
        print(
            "⚠️  No model artifacts found in backend/artifacts/. "
            "Run `python train.py` after placing data/train.csv."
        )


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": asd_model.loaded}


def _build_message(risk: str, aq_score: int) -> str:
    if risk == "high":
        return (
            f"Based on an AQ-10 score of {aq_score}/10 and the responses provided, "
            "the model indicates traits associated with a higher likelihood of ASD. "
            "This is a screening signal only — please consult a qualified healthcare "
            "professional for a full evaluation."
        )
    return (
        f"Based on an AQ-10 score of {aq_score}/10 and the responses provided, "
        "the model indicates a lower likelihood of ASD traits. "
        "If you still have concerns, consider speaking with a healthcare professional."
    )


@app.post("/predict", response_model=ASDPrediction)
def predict(payload: ASDInput):
    try:
        prediction, probability = asd_model.predict(payload.model_dump())
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    risk = "high" if prediction == 1 else "low"
    label = "Higher Likelihood of ASD Traits" if risk == "high" else "Lower Likelihood of ASD Traits"
    aq_score = int(round(payload.result))

    return ASDPrediction(
        prediction=prediction,
        risk=risk,
        label=label,
        message=_build_message(risk, aq_score),
        confidence=round(probability, 4),
        aq_score=aq_score,
    )
