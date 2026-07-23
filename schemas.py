import uuid

from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    text: str

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    probability: float
    entropy: float
    attention: float
    opinion_word: str

class PredictResponse(BaseModel):
    id: uuid.UUID = None
    original_text: str
    message: str
    results: List[AspectResult]

class ReviewItem(BaseModel):
    id: uuid.UUID
    text: str

class BatchPredictRequest(BaseModel):
    reviews: List[ReviewItem]

class BatchPredictResponse(BaseModel):
    message: str
    total_processed: int
    batch_results: List[PredictResponse]