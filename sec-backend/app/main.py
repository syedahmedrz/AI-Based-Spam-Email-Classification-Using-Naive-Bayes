from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib  # For loading the saved model
import os

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
orig_origins = [
    "http://localhost:3000",  # React app URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orig_origins,  # Allows all origins specified above
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), "saved_model/spam_classifier_model.pkl")
model = joblib.load(model_path)

# Load the vectorizer
vectorizer_path = os.path.join(os.path.dirname(__file__), "saved_model/vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# Root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Define the input structure using Pydantic
class EmailRequest(BaseModel):
    email_subject: str  # Defines that the input should be a string

# POST endpoint to make predictions
@app.post("/predict/")
def predict_email_spam(email_request: EmailRequest):
    email_subject = email_request.email_subject  # Get the email subject from the request
    email_subject_vectorized = vectorizer.transform([email_subject])  # Vectorize the email subject

    # Perform classification using the loaded model
    prediction = model.predict(email_subject_vectorized)
    
    # Return the prediction result
    return {"prediction": prediction[0]}
