"""
Sentiment Analysis Classifier for Hotel Reviews
Wraps the HotelReviewNLPModel for use with FastAPI backend
"""

import os
import sys
import pickle
from pathlib import Path

# Add Project4 directory to path so imports work correctly
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "trained_nlp_model.pkl")


class SentimentClassifier:
    """Wrapper for the trained NLP sentiment analysis model"""

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained sentiment model from disk"""
        if os.path.exists(MODEL_PATH):
            try:
                # Ensure the Project4 directory is in the path for pickle unpickling
                if BASE_DIR not in sys.path:
                    sys.path.insert(0, BASE_DIR)

                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Sentiment model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print(f"   Make sure sentiment_scorer.py, data_preprocessor.py, and ner_model.py are in {BASE_DIR}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}")
            self.model = None

    def classify(self, review_text):
        """
        Classify the sentiment of a hotel review

        Args:
            review_text (str): The hotel review text to analyze

        Returns:
            dict: Classification result with prediction, confidence, and sentiment scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot classify review.")

        try:
            # Check if model is a dict (from pickle) or an object
            if isinstance(self.model, dict):
                # Model was saved as a dictionary of components
                # Use the components directly
                sentiment_scorer = self.model.get('sentiment_scorer')
                tfidf_vectorizer = self.model.get('tfidf_vectorizer')
                logistic_model = self.model.get('logistic_model')

                if not all([sentiment_scorer, tfidf_vectorizer, logistic_model]):
                    raise RuntimeError("Model components missing from pickle file")

                # Import preprocessing function
                from data_preprocessor import preprocess_text

                # Preprocess the text
                processed_text = preprocess_text(review_text, remove_punct=True, remove_stops=True, keep_negations=True)

                # Vectorize
                text_tfidf = tfidf_vectorizer.transform([processed_text])

                # Predict
                prediction = logistic_model.predict(text_tfidf)[0]
                probability = logistic_model.predict_proba(text_tfidf)[0]

                # Get sentiment score
                sentiment_result = sentiment_scorer.score_text(processed_text)

                # Determine classification and confidence
                classification = 'Positive' if prediction == 1 else 'Negative'
                # Confidence should be the probability of the PREDICTED class
                confidence = float(probability[1]) if prediction == 1 else float(probability[0])

                # Map classification to star rating based on confidence
                # Negative (high confidence) -> 1-2 stars
                # Negative (low confidence) -> 2-3 stars
                # Positive (low confidence) -> 3-4 stars
                # Positive (high confidence) -> 4-5 stars
                if prediction == 0:  # Negative
                    # Map confidence (0.5-1.0) to star range (1.0-2.5)
                    # Higher confidence in negative = lower stars
                    sentiment_score = 2.5 - (confidence - 0.5) * 3.0
                else:  # Positive
                    # Map confidence (0.5-1.0) to star range (2.5-5.0)
                    # Higher confidence in positive = higher stars
                    sentiment_score = 2.5 + (confidence - 0.5) * 5.0

                # Determine sentiment label based on score
                if sentiment_score >= 4.5:
                    sentiment_label = 'Very Positive'
                elif sentiment_score >= 3.5:
                    sentiment_label = 'Positive'
                elif sentiment_score >= 2.5:
                    sentiment_label = 'Neutral'
                elif sentiment_score >= 1.5:
                    sentiment_label = 'Negative'
                else:
                    sentiment_label = 'Very Negative'

                return {
                    "original_text": review_text,
                    "processed_text": processed_text,
                    "classification": classification,
                    "confidence": confidence,  # Confidence in the PREDICTED class
                    "positive_probability": float(probability[1]),
                    "negative_probability": float(probability[0]),
                    "sentiment_score": float(sentiment_score),  # Mapped from classification (1-5)
                    "sentiment_label": sentiment_label,  # Mapped from score
                    "success": True
                }
            else:
                # Model is a HotelReviewNLPModel object
                result = self.model.predict_sentiment(review_text)

                # Determine confidence based on actual prediction
                classification = result['prediction_label']
                confidence = float(result['positive_probability']) if classification == 'Positive' else float(result['negative_probability'])

                # Map classification to star rating based on confidence
                # Negative (high confidence) -> 1-2 stars
                # Negative (low confidence) -> 2-3 stars
                # Positive (low confidence) -> 3-4 stars
                # Positive (high confidence) -> 4-5 stars
                if classification == 'Negative':
                    # Map confidence (0.5-1.0) to star range (1.0-2.5)
                    # Higher confidence in negative = lower stars
                    sentiment_score = 2.5 - (confidence - 0.5) * 3.0
                else:  # Positive
                    # Map confidence (0.5-1.0) to star range (2.5-5.0)
                    # Higher confidence in positive = higher stars
                    sentiment_score = 2.5 + (confidence - 0.5) * 5.0

                # Determine sentiment label based on score
                if sentiment_score >= 4.5:
                    sentiment_label = 'Very Positive'
                elif sentiment_score >= 3.5:
                    sentiment_label = 'Positive'
                elif sentiment_score >= 2.5:
                    sentiment_label = 'Neutral'
                elif sentiment_score >= 1.5:
                    sentiment_label = 'Negative'
                else:
                    sentiment_label = 'Very Negative'

                # Format result for API response
                return {
                    "original_text": result['original_text'],
                    "processed_text": result['processed_text'],
                    "classification": classification,
                    "confidence": confidence,  # Confidence in the PREDICTED class
                    "positive_probability": float(result['positive_probability']),
                    "negative_probability": float(result['negative_probability']),
                    "sentiment_score": float(sentiment_score),  # Mapped from classification (1-5)
                    "sentiment_label": sentiment_label,  # Mapped from score
                    "success": True
                }
        except Exception as e:
            raise RuntimeError(f"Error during classification: {e}")


# Global classifier instance
_classifier = None


def get_classifier():
    """Get or initialize the global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = SentimentClassifier()
    return _classifier
