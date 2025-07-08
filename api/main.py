# FastAPI framework for building the REST API
from fastapi import FastAPI, HTTPException  # Core API framework and error handling

# Pydantic for request/response data validation
from pydantic import BaseModel  # For defining request/response schemas

# Typing for type hints (optional, but good practice)
from typing import List  # For type annotations in endpoints

# Hugging Face Transformers for loading the tokenizer and model
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification  # For model/tokenizer loading

# PyTorch for tensor operations and model inference
import torch  # For running inference on the model

# For loading the model from the correct path
import os  # For file path management

# For logging API progress and requests
import logging  # For logging API activity

# --- FastAPI App Initialization ---

# Initialize FastAPI app
app = FastAPI(title="Fake Job Posting Detection API", description="API for DeBERTa-based fake job posting classification", version="1.0")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("FastAPI app initialized and logging is set up.")

# --- Model and Tokenizer Loading ---

# Path to the best model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model/model/deberta_best_model')
MODEL_NAME = 'microsoft/deberta-v3-base'

def load_model_and_tokenizer():
    try:
        logger.info(f"Loading tokenizer from {MODEL_NAME}...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        logger.info(f"Loading model from {MODEL_DIR}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        raise

# Load at startup
tokenizer, model = load_model_and_tokenizer()

# --- Request and Response Schemas ---

class JobPostingRequest(BaseModel):
    """Request schema for job posting prediction."""
    text: str  # The job posting text (description + requirements)

class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    label: str  # 'fraudulent' or 'legitimate'
    probability: float  # Probability of the predicted label

# --- Progress Log ---
# [x] Imported all necessary libraries for FastAPI inference API, with clear comments for each.
# [x] Initialized FastAPI app and set up logging.
# [x] Loaded DeBERTa model and tokenizer at startup, with error handling and logging.
# [x] Defined Pydantic request and response schemas for the API.
# [x] Implemented /predict endpoint for job posting inference, with logging and error handling. 

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint for API health check and welcome message.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Fake Job Posting Detection API! Use /predict to get predictions."}

@app.post("/predict", response_model=PredictionResponse)
def predict_job_posting(request: JobPostingRequest):
    """Predict if a job posting is fraudulent or legitimate."""
    logger.info("Received prediction request.")
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx = int(probabilities.argmax())
            pred_prob = float(probabilities[pred_idx])
        label = 'fraudulent' if pred_idx == 1 else 'legitimate'
        logger.info(f"Prediction: {label} (probability: {pred_prob:.4f})")
        return PredictionResponse(label=label, probability=pred_prob)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.") 