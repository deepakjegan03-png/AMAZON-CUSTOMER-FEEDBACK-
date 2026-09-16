Act as a senior Python Full-Stack Developer and Machine Learning Engineer.

I have a trained Spam Message Classification Machine Learning model saved as a Pickle file. The model is built using Scikit-learn's Multinomial Naive Bayes algorithm.

Your task is to build a complete, professional, and locally runnable web application using FastAPI for the backend and HTML, CSS, and JavaScript for the frontend.

## Project Details

Project Name: Spam Message Classification Web Application

Machine Learning Model:

* Algorithm: Multinomial Naive Bayes
* Framework: Scikit-learn
* Model File: spam_classifier.pkl
* Task: Classify user-entered messages as Spam or Not Spam (Ham).

The application must load the trained Pickle model and use it to predict whether a message is spam.

## Important Machine Learning Requirement

The uploaded Pickle file contains a trained MultinomialNB classifier.

Inspect the model carefully and create the correct prediction pipeline.

The application must also support loading the corresponding text vectorizer used during training, such as:

* vectorizer.pkl
* tfidf_vectorizer.pkl
* count_vectorizer.pkl

If the vectorizer is not available, clearly identify this requirement and structure the application so that the vectorizer can be added easily without changing the frontend.

Do not retrain the model unless absolutely necessary.

## Technology Stack

Backend:

* Python 3.10+
* FastAPI
* Uvicorn
* Scikit-learn
* Joblib
* Pydantic

Frontend:

* HTML5
* CSS3
* JavaScript
* Jinja2 Templates

UI Framework:

* Use modern responsive CSS.
* Create a clean, premium, professional Machine Learning dashboard.

## Required Project Structure

Create the following folder structure:

spam_classification_app/
│
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── spam_classifier.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── src/
├── **init**.py
├── model_loader.py
└── predictor.py

Keep the project modular and maintainable.

## Backend Requirements

Create a FastAPI application with the following features:

1. Load the Pickle Machine Learning model when the application starts.

2. Load the text vectorizer separately.

3. Create a prediction function that:

   * Accepts raw text.
   * Applies the same preprocessing/vectorization used during training.
   * Sends the transformed text to the trained model.
   * Returns the prediction.

4. Create API endpoints:

GET /

* Display the frontend web page.

POST /predict

* Accept JSON input containing a message.
* Return prediction result.

Expected JSON Request:

{
"message": "Congratulations! You won a free lottery ticket."
}

Expected JSON Response:

{
"message": "Congratulations! You won a free lottery ticket.",
"prediction": "Spam",
"confidence": 0.98
}

Use appropriate error handling for:

* Empty messages.
* Missing model files.
* Invalid input.
* Server errors.

## Frontend Design Requirements

Create a beautiful and modern Spam Detection Dashboard.

Design Style:

* Premium dark and light professional theme.
* Clean typography.
* Responsive design for desktop and mobile.
* Modern cards and buttons.
* Smooth animations.
* Professional Data Science / AI dashboard appearance.

The page should contain:

### Header

* Application name: SpamGuard AI
* Subtitle: Intelligent Message Classification System
* Small badge: Powered by Machine Learning

### Main Section

Create a large message input textarea.

Placeholder:
"Enter your email or SMS message here to check whether it is spam..."

Add a prominent button:

"Analyze Message"

### Prediction Result Section

After clicking Analyze Message, display:

* Prediction: SPAM or NOT SPAM
* Confidence Score
* Visual status card
* Explanation message

For Spam:
"Warning: This message appears to be potentially spam."

For Ham:
"This message appears to be legitimate."

### Additional UI Features

Include:

* Character counter.
* Clear button.
* Loading animation while prediction is running.
* Error message display.
* Color-coded prediction result.
* Confidence percentage progress bar.
* Example message buttons for testing.

Example test messages:

Spam:
"Congratulations! You have won a free iPhone. Click here to claim your prize."

Ham:
"Hi Deepak, please attend the project meeting tomorrow at 10 AM."

## API Integration

Use JavaScript Fetch API to communicate with FastAPI.

The frontend should send requests to:

POST /predict

Do not reload the page when predicting.

Display results dynamically.

## Security and Validation

Implement:

* Pydantic request validation.
* Maximum message length validation.
* Proper exception handling.
* No hardcoded predictions.
* Never expose model internals unnecessarily.

## Requirements File

Create requirements.txt containing all necessary dependencies, such as:

fastapi
uvicorn
scikit-learn
joblib
jinja2
python-multipart

Use compatible versions where necessary.

## Localhost Running Instructions

The application must run locally on Windows using PowerShell or Command Prompt.

Provide exact commands:

1. Create virtual environment:

python -m venv .venv

2. Activate environment:

.venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Run FastAPI application:

uvicorn app:app --reload

5. Open browser:

http://127.0.0.1:8000

## README Requirements

Create a complete README.md containing:

* Project overview.
* Features.
* Technologies used.
* Project structure.
* Installation instructions.
* How to run locally.
* API endpoint documentation.
* Sample JSON request and response.
* Machine Learning model explanation.
* Future enhancements.

## Quality Requirements

Write complete production-style code.

Do not provide incomplete snippets.

Ensure:

* All imports are correct.
* File paths work on Windows.
* FastAPI templates load correctly.
* Static CSS and JavaScript files work correctly.
* The Pickle model loads correctly.
* API prediction works without errors.
* The application is beginner-friendly to understand.

Before finalizing, verify the complete project structure and provide all necessary files with their full code.

The final application should be ready to run on localhost.
