# Complaint Category Predictor

A machine learning-based web application that classifies customer complaints into categories and assigns priority levels.

## Features

- **Complaint Classification**: Automatically categorizes complaints into:
  - Payment
  - Login
  - Delivery
  - Technical
  - Other

- **Priority Assignment**: Automatically assigns priority levels:
  - High: Payment and Login issues
  - Medium: Delivery and Technical issues
  - Low: Other complaints

- **User-Friendly Interface**: Clean, modern web interface with example complaints

- **Confidence Scoring**: Shows prediction confidence percentage

## Project Structure

```
task/
├── app.py                 # FastAPI backend
├── train_model.py         # ML model training script
├── complaint_classifier.pkl # Trained model
├── index.html            # Frontend interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already downloaded):
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   ```

4. **Train the model** (optional - pre-trained model included):
   ```bash
   python train_model.py
   ```

## Usage

### Starting the Backend Server

```bash
python app.py
```

The server will start at `http://localhost:8000`

### Accessing the Frontend

Open `index.html` in your web browser or navigate to `http://localhost:8000` after starting the server.

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict complaint category
- `GET /categories` - Get all categories
- `GET /priority-info` - Get priority information

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"complaint": "My payment was deducted but my order was not confirmed."}'
```

### Example Response

```json
{
  "category": "Payment",
  "priority": "High",
  "confidence": 0.85
}
```

## How It Works

1. **Text Preprocessing**: Complaints are cleaned, tokenized, and stopwords are removed
2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
3. **Classification**: Naive Bayes classifier predicts the category
4. **Priority Mapping**: Category determines priority level

## Model Performance

The model achieves approximately 62% accuracy on the sample dataset. For production use, you would want to train on a larger, more diverse dataset.

## Testing the Application

1. Start the backend server: `python app.py`
2. Open `index.html` in your browser
3. Enter a complaint or use the example buttons
4. Click "Analyze Complaint" to see results

## Example Complaints to Try

- "My payment was deducted but my order was not confirmed." → Payment, High
- "I cannot login to my account." → Login, High
- "My order has not been delivered yet." → Delivery, Medium
- "Website is not loading properly." → Technical, Medium
- "Customer service is not responding." → Other, Low

## Technologies Used

- **Backend**: FastAPI, Python
- **Machine Learning**: scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Model Persistence**: joblib

## License

This project is for demonstration purposes.