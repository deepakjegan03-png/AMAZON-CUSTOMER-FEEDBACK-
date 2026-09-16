import re
import logging
from typing import Dict, Any, Optional
import spacy

logger = logging.getLogger("spam_guard.predictor")

# Lazy-load spaCy model
_nlp = None


def get_nlp():
    """Lazily load the spaCy English model with disabled NER and parser for maximum speed."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("Loaded spaCy en_core_web_sm model")
        except Exception as e:
            logger.warning(
                "Could not load spacy en_core_web_sm (%s). Falling back to basic regex cleaning.",
                e,
            )
            _nlp = False
    return _nlp


def clean_text(text: str) -> str:
    """
    Clean and preprocess raw text to match the exact training pipeline:
    1. Lowercase conversion
    2. Strip URLs
    3. Remove non-alphabetical characters
    4. Lemmatize words and drop stop words using spaCy
    """
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    nlp = get_nlp()
    if nlp:
        doc = nlp(text)
        words = [t.lemma_ for t in doc if not t.is_stop and t.lemma_.strip()]
        cleaned = " ".join(words)
    else:
        # Fallback if spaCy is unavailable
        cleaned = " ".join(text.split())

    return cleaned


class SpamPredictor:
    """
    Spam Message Predictor wrapper encapsulating the loaded model,
    vectorizer, and preprocessing pipeline.
    """

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.classes = list(getattr(model, "classes_", ["ham", "spam"]))

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Execute full prediction pipeline:
        1. Preprocess raw message
        2. Vectorize with trained vectorizer
        3. Predict class and class probabilities with MultinomialNB
        4. Package result with confidence and descriptive metadata
        """
        if not raw_message or not raw_message.strip():
            raise ValueError("Input message cannot be empty or whitespace.")

        cleaned = clean_text(raw_message)

        # In edge cases where cleaning removes all tokens (e.g., purely punctuation),
        # use a fallback on stripped lower text so vectorizer doesn't receive empty token space.
        text_for_vectorizer = cleaned if cleaned.strip() else raw_message.lower().strip()

        X = self.vectorizer.transform([text_for_vectorizer])

        # Get discrete class prediction
        pred_label = str(self.model.predict(X)[0]).lower()

        # Compute probabilities if supported by the model
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            prob_dict = {
                str(cls).lower(): round(float(prob), 4)
                for cls, prob in zip(self.classes, probs)
            }
            # Confidence is the probability of the predicted class
            pred_idx = [str(c).lower() for c in self.classes].index(pred_label)
            confidence = float(probs[pred_idx])
        else:
            confidence = 1.0
            prob_dict = {pred_label: 1.0}

        is_spam = pred_label == "spam"
        display_prediction = "Spam" if is_spam else "Not Spam"
        explanation = (
            "Warning: This message appears to be potentially spam."
            if is_spam
            else "This message appears to be legitimate."
        )

        return {
            "message": raw_message,
            "prediction": display_prediction,
            "label": pred_label,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "cleaned_text": cleaned,
            "explanation": explanation,
        }
