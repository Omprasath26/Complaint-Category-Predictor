from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Initialize FastAPI app
app = FastAPI(title="Complaint Category Predictor API", 
              description="API for classifying customer complaints into categories",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Load the trained model
try:
    model = joblib.load('complaint_classifier.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define request and response models
class ComplaintRequest(BaseModel):
    complaint: str

class PredictionResponse(BaseModel):
    category: str
    priority: str
    confidence: float = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text by removing special characters, converting to lowercase, and removing stopwords"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back
    return ' '.join(tokens)

# Priority mapping
priority_map = {
    'Payment': 'High',
    'Login': 'High',
    'Delivery': 'Medium',
    'Technical': 'Medium',
    'Other': 'Low'
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML interface"""
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    else:
        # Fallback to API info if frontend not found
        return {
            "message": "Complaint Category Predictor API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "docs": "/docs"
            }
        }

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Complaint Category Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_complaint(request: ComplaintRequest):
    """Predict category and priority for a complaint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.complaint or request.complaint.strip() == "":
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty")
    
    try:
        # Preprocess the complaint text
        processed_text = preprocess_text(request.complaint)
        
        # Predict category
        category = model.predict([processed_text])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba([processed_text])[0]
        category_index = list(model.classes_).index(category)
        confidence = float(probabilities[category_index])
        
        # Determine priority
        priority = priority_map.get(category, 'Low')
        
        return PredictionResponse(
            category=category,
            priority=priority,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get all available complaint categories"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    categories = list(model.classes_)
    return {
        "categories": categories,
        "count": len(categories)
    }

@app.get("/priority-info")
async def get_priority_info():
    """Get information about priority levels"""
    return {
        "priority_mapping": priority_map,
        "description": {
            "High": "Requires immediate attention (Payment, Login issues)",
            "Medium": "Important but not urgent (Delivery, Technical issues)",
            "Low": "General inquiries and other complaints"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)