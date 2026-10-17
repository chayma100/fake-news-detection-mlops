from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    articles: list[str] = Field(..., min_length=1, max_length=100)

class Prediction(BaseModel):
    label: str            # "fake" ou "real"
    probability: float

class PredictResponse(BaseModel):
    predictions: list[Prediction]