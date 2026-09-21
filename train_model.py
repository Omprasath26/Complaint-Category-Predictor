import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Create sample training data based on the problem statement
def create_training_data():
    """Create a sample dataset for complaint classification"""
    
    # Sample complaints for each category
    complaints = [
        # Payment complaints
        ("My payment was deducted but my order was not confirmed.", "Payment"),
        ("I was charged twice for the same order.", "Payment"),
        ("Payment failed but money was deducted from my account.", "Payment"),
        ("Refund not received after cancellation.", "Payment"),
        ("Payment gateway is not working.", "Payment"),
        ("Credit card payment declined.", "Payment"),
        ("Unable to make payment through UPI.", "Payment"),
        ("Payment processed but order status shows pending.", "Payment"),
        
        # Login complaints
        ("I cannot login to my account.", "Login"),
        ("Forgot password option not working.", "Login"),
        ("Account locked after multiple failed attempts.", "Login"),
        ("Two-factor authentication not working.", "Login"),
        ("Unable to reset password.", "Login"),
        ("Login page keeps loading.", "Login"),
        ("Session expires too quickly.", "Login"),
        ("Wrong password error even with correct password.", "Login"),
        
        # Delivery complaints
        ("My order has not been delivered yet.", "Delivery"),
        ("Package delivered to wrong address.", "Delivery"),
        ("Delivery delayed by 3 days.", "Delivery"),
        ("Item damaged during delivery.", "Delivery"),
        ("Tracking information not updated.", "Delivery"),
        ("Delivery person didn't call before delivery.", "Delivery"),
        ("Wrong item delivered.", "Delivery"),
        ("Delivery charge is too high.", "Delivery"),
        
        # Technical complaints
        ("Website is not loading properly.", "Technical"),
        ("App crashes when I try to checkout.", "Technical"),
        ("Images not loading on product pages.", "Technical"),
        ("Search function not working.", "Technical"),
        ("Page keeps refreshing automatically.", "Technical"),
        ("Cannot add items to cart.", "Technical"),
        ("Checkout process is stuck.", "Technical"),
        ("Error 500 when accessing my account.", "Technical"),
        
        # Other complaints
        ("Customer service is not responding.", "Other"),
        ("Need to update my contact information.", "Other"),
        ("Want to cancel my subscription.", "Other"),
        ("Feedback about the service.", "Other"),
        ("General inquiry about products.", "Other"),
        ("Complaint about pricing.", "Other"),
        ("Suggestion for improvement.", "Other"),
        ("Request for partnership.", "Other")]
    
    # Convert to DataFrame
    df = pd.DataFrame(complaints, columns=['complaint', 'category'])
    return df

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

def train_model():
    """Train the complaint classification model"""
    print("Creating training data...")
    df = create_training_data()
    
    print("Preprocessing text data...")
    df['processed_complaint'] = df['complaint'].apply(preprocess_text)
    
    # Split data
    X = df['processed_complaint']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Create pipeline with TF-IDF vectorizer and Naive Bayes classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', MultinomialNB())
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'complaint_classifier.pkl')
    print("\nModel saved as 'complaint_classifier.pkl'")
    
    # Also save the vectorizer and classifier separately for easier access
    joblib.dump(model.named_steps['tfidf'], 'tfidf_vectorizer.pkl')
    joblib.dump(model.named_steps['clf'], 'naive_bayes_classifier.pkl')
    
    return model

def predict_category(complaint_text, model=None):
    """Predict category for a single complaint"""
    if model is None:
        model = joblib.load('complaint_classifier.pkl')
    
    # Preprocess the text
    processed_text = preprocess_text(complaint_text)
    
    # Predict category
    category = model.predict([processed_text])[0]
    
    # Determine priority based on category
    priority_map = {
        'Payment': 'High',
        'Login': 'High',
        'Delivery': 'Medium',
        'Technical': 'Medium',
        'Other': 'Low'
    }
    
    priority = priority_map.get(category, 'Low')
    
    return category, priority

if __name__ == "__main__":
    # Train and save the model
    model = train_model()
    
    # Test with the example from problem statement
    test_complaint = "My payment was deducted but my order was not confirmed."
    category, priority = predict_category(test_complaint, model)
    
    print(f"\nTest Complaint: {test_complaint}")
    print(f"Predicted Category: {category}")
    print(f"Priority: {priority}")
    
    # Test with a few more examples
    test_cases = [
        "I cannot login to my account.",
        "My order has not been delivered yet.",
        "Website is not loading properly.",
        "Customer service is not responding."
    ]
    
    print("\nAdditional Test Cases:")
    for complaint in test_cases:
        category, priority = predict_category(complaint, model)
        print(f"Complaint: {complaint}")
        print(f"  Category: {category}, Priority: {priority}")