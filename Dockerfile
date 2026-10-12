
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

COPY app.py .
COPY complaint_classifier.pkl .
COPY index.html .

EXPOSE 8000

CMD ["python", "app.py"]
