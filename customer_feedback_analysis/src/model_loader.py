import os
import logging
import joblib

logger = logging.getLogger("spam_guard.model_loader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Default model candidates in priority order
MODEL_CANDIDATES = [
    "spam_classifier.pkl",
    "spam_model.pkl",
]

# Default vectorizer candidates in priority order
VECTORIZER_CANDIDATES = [
    "vectorizer.pkl",
    "count_vectorizer.pkl",
    "tfidf_vectorizer.pkl",
]


def find_file(candidates, directory=MODELS_DIR):
    """Find the first existing file among candidates in the given directory."""
    for candidate in candidates:
        full_path = os.path.join(directory, candidate)
        if os.path.exists(full_path):
            return full_path
    return None


def load_model(model_path: str = None):
    """
    Load the trained machine learning model from pickle.
    Raises FileNotFoundError if model file cannot be located.
    """
    if model_path is None:
        model_path = find_file(MODEL_CANDIDATES)

    if not model_path or not os.path.exists(model_path):
        candidates_str = ", ".join(MODEL_CANDIDATES)
        raise FileNotFoundError(
            f"Spam classification model not found. Looked for [{candidates_str}] in {MODELS_DIR}. "
            f"Please ensure spam_classifier.pkl is placed in the models directory."
        )

    try:
        model = joblib.load(model_path)
        logger.info("Successfully loaded model from %s", model_path)
        return model, model_path
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", model_path, exc)
        raise RuntimeError(f"Error loading model from {model_path}: {str(exc)}") from exc


def load_vectorizer(vectorizer_path: str = None):
    """
    Load the trained text vectorizer from pickle.
    Raises FileNotFoundError if vectorizer file cannot be located.
    """
    if vectorizer_path is None:
        vectorizer_path = find_file(VECTORIZER_CANDIDATES)

    if not vectorizer_path or not os.path.exists(vectorizer_path):
        candidates_str = ", ".join(VECTORIZER_CANDIDATES)
        raise FileNotFoundError(
            f"Text vectorizer not found. Looked for [{candidates_str}] in {MODELS_DIR}. "
            f"Please ensure vectorizer.pkl is placed in the models directory."
        )

    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Successfully loaded vectorizer from %s", vectorizer_path)
        return vectorizer, vectorizer_path
    except Exception as exc:
        logger.error("Failed to load vectorizer from %s: %s", vectorizer_path, exc)
        raise RuntimeError(f"Error loading vectorizer from {vectorizer_path}: {str(exc)}") from exc


def load_pipeline_components():
    """
    Load both model and vectorizer together.
    Returns:
        dict: {'model': model, 'vectorizer': vectorizer, 'model_path': path, 'vectorizer_path': path}
    """
    model, m_path = load_model()
    vectorizer, v_path = load_vectorizer()
    return {
        "model": model,
        "vectorizer": vectorizer,
        "model_path": m_path,
        "vectorizer_path": v_path,
    }
