# SpamGuard AI — Intelligent Message Classification Web Application

A full-stack, machine-learning-powered web application built with **FastAPI**, **HTML5**, **CSS3**, and **JavaScript** that classifies email and SMS messages into **Spam** or **Not Spam (Legitimate / Ham)** in real time.

---

## 📌 Project Overview

SpamGuard AI integrates a trained Scikit-learn **Multinomial Naive Bayes** classifier with a high-performance **CountVectorizer** text representation pipeline. Users can input any message via an interactive, responsive dashboard to receive instant predictions, confidence metrics, class probabilities, and preprocessed NLP token breakdowns.

---

## ✨ Features

- **Real-Time Classification**: Instant classification of user-entered text using a trained MultinomialNB model.
- **Confidence & Probability Metrics**: Displays the exact model confidence score alongside an animated progress bar and full class probability breakdown (`Spam` vs. `Legitimate`).
- **Interactive Web Dashboard**:
  - Premium Dark and Light mode themes with local preference persistence.
  - Test sample buttons (pre-loaded with representative Spam and Ham messages).
  - Character counter with a 5,000-character safety limit and clear button.
  - Loading animation during server inference.
  - Transparent NLP token viewer showing lemmatized, stopword-filtered tokens.
- **Resilient Model Loader**:
  - Automatically searches for model files (`spam_classifier.pkl` / `spam_model.pkl`) and vectorizers (`vectorizer.pkl` / `count_vectorizer.pkl`).
  - Clear, user-friendly exception handling if assets are missing.
- **RESTful API**:
  - `GET /` — Renders dashboard via Jinja2 templates.
  - `POST /predict` — JSON classification endpoint with Pydantic schema validation.
  - `GET /health` — Diagnostic health check reporting pipeline status.

---

## 🛠️ Technologies Used

### Backend
- **Python 3.10+**
- **FastAPI**: Modern, high-performance web framework for building APIs.
- **Uvicorn**: Lightning-fast ASGI server.
- **Scikit-learn**: Machine learning algorithms (`MultinomialNB`, `CountVectorizer`).
- **Joblib**: Efficient serialization for model and vectorizer persistence.
- **SpaCy (`en_core_web_sm`)**: Industrial-strength NLP preprocessing and lemmatization.
- **Pydantic v2**: Request data validation and response schemas.

### Frontend
- **HTML5 & Jinja2**: Semantic templating and structured layout.
- **Modern CSS3**: Responsive design, custom variables, smooth transitions, glassmorphism.
- **Vanilla JavaScript (ES6+)**: Dynamic DOM manipulation, Fetch API, clipboard interaction.

---

## 📂 Project Structure

```
customer_feedback_analysis/
├── app.py                     # FastAPI application entrypoint & API routes
├── requirements.txt           # Project dependencies
├── README.md                  # Complete project documentation
├── prompt.md                  # Project specification prompt
│
├── models/                    # Serialized machine learning assets
│   ├── spam_classifier.pkl    # Trained MultinomialNB classifier
│   └── vectorizer.pkl         # Trained CountVectorizer
│
├── src/                       # Application logic & ML pipeline
│   ├── __init__.py
│   ├── model_loader.py        # Safe loader for model and vectorizer
│   └── predictor.py           # NLP cleaning, vectorization & inference
│
├── templates/
│   └── index.html             # Dashboard Jinja2 HTML template
│
└── static/
    ├── style.css              # Custom styling (Dark / Light mode)
    └── script.js              # Client-side logic & API integration
```

---

## 🚀 Installation & Local Setup

The application is fully runnable locally on Windows using PowerShell or Command Prompt.

### 1. Clone or Open the Project
Open terminal in the project directory:
```powershell
cd "c:\Users\acer\OneDrive\Documents - Copy\customer_feedback_analysis"
```

### 2. Create Virtual Environment (Optional if `.venv` exists)
```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment
- **PowerShell**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Command Prompt**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Download the spaCy Model (if not already installed)
```powershell
python -m spacy download en_core_web_sm
```

### 6. Start the FastAPI Server
```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 7. Open Your Browser
Navigate to:
```
http://127.0.0.1:8000
```

Interactive API documentation is also available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 📡 API Documentation

### 1. Classify Message (`POST /predict`)

**Endpoint**: `POST /predict`  
**Content-Type**: `application/json`

#### Sample Request:
```json
{
  "message": "Congratulations! You have won a free iPhone. Click here to claim your prize."
}
```

#### Sample Response:
```json
{
  "message": "Congratulations! You have won a free iPhone. Click here to claim your prize.",
  "prediction": "Spam",
  "label": "spam",
  "is_spam": true,
  "confidence": 1.0,
  "confidence_percentage": 100.0,
  "probabilities": {
    "ham": 0.0,
    "spam": 1.0
  },
  "cleaned_text": "congratulation win free iphone click claim prize",
  "explanation": "Warning: This message appears to be potentially spam."
}
```

---

### 2. Health Check (`GET /health`)

**Endpoint**: `GET /health`

#### Sample Response:
```json
{
  "status": "healthy",
  "pipeline": {
    "status": "ready",
    "model_path": "models/spam_classifier.pkl",
    "vectorizer_path": "models/vectorizer.pkl",
    "error": null
  }
}
```

---

## 🧠 Machine Learning Architecture

1. **Text Preprocessing**:
   - Lowercases input string.
   - Strips URLs and hyper-links (`http\S+`).
   - Removes punctuation, symbols, and non-alphabetical characters.
   - Feeds tokens to spaCy's `en_core_web_sm` pipeline to extract base lemmas while dropping common English stop-words.
2. **Text Vectorization**:
   - Transformed by the trained `CountVectorizer` into a bag-of-words token count matrix.
3. **Classification & Inference**:
   - Evaluated by `MultinomialNB` to generate posterior probability distribution $P(C_k \mid X)$.
   - Outputs the predicted class (`spam` or `ham`) and normalized confidence score.

---

## 🔮 Future Enhancements

- Support for batch classification via CSV/Excel drag-and-drop.
- Real-time explainability highlighting words that contributed most to the spam decision.
- Email server webhook integration (IMAP/Gmail API) for automated mailbox screening.
- Dockerized deployment with container health checks.
