from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """
    input_msg: str = Field(..., example="este es un mensaje", title='message to predict if is SPAM o HAM')


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """
    result: str = Field(..., example="SPAM", title='Probablity for class SPAM')


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')