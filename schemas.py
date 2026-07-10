"""
schemas.py
----------
Pydantic models for the /predict endpoint request and response.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ASDInput(BaseModel):
    A1_Score: int = Field(..., ge=0, le=1)
    A2_Score: int = Field(..., ge=0, le=1)
    A3_Score: int = Field(..., ge=0, le=1)
    A4_Score: int = Field(..., ge=0, le=1)
    A5_Score: int = Field(..., ge=0, le=1)
    A6_Score: int = Field(..., ge=0, le=1)
    A7_Score: int = Field(..., ge=0, le=1)
    A8_Score: int = Field(..., ge=0, le=1)
    A9_Score: int = Field(..., ge=0, le=1)
    A10_Score: int = Field(..., ge=0, le=1)
    age: int = Field(..., ge=1, le=120)
    gender: Literal["m", "f"]
    ethnicity: str
    jaundice: Literal["yes", "no"]
    austim: Literal["yes", "no"]
    contry_of_res: str
    used_app_before: Literal["yes", "no"]
    result: float
    relation: str

    class Config:
        json_schema_extra = {
            "example": {
                "A1_Score": 1, "A2_Score": 0, "A3_Score": 1, "A4_Score": 0, "A5_Score": 1,
                "A6_Score": 0, "A7_Score": 1, "A8_Score": 1, "A9_Score": 0, "A10_Score": 1,
                "age": 25, "gender": "f", "ethnicity": "White-European",
                "jaundice": "no", "austim": "no", "contry_of_res": "United States",
                "used_app_before": "no", "result": 6, "relation": "Self",
            }
        }


class ASDPrediction(BaseModel):
    prediction: int
    risk: Literal["low", "high"]
    label: str
    message: str
    confidence: float
    aq_score: int
