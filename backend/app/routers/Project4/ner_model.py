"""
Named Entity Recognition and Sentiment Analysis Model
Implements a complete NLP pipeline with:
- Text preprocessing (imported from data_preprocessor.py - no duplication)
- Sentiment scoring using TF-IDF (imported from sentiment_scorer.py)
- Binary classification using Logistic Regression
- 80:20 train/test split

This module orchestrates the complete pipeline by importing and using
functions from data_preprocessor.py and sentiment_scorer.py rather than
redefining them, following DRY (Don't Repeat Yourself) principles.
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Preprocessing and text analysis (minimal imports - most functions imported from data_preprocessor)
import nltk

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom classes and functions from existing files
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment_scorer import SentimentScorer
from data_preprocessor import (
    download_nltk_resources,
    remove_punctuation,
    remove_stopwords,
    preprocess_text,
    clean_text
)


class HotelReviewNLPModel:
    """
    Complete NLP model for hotel review sentiment analysis and classification
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the NLP model
        
        Args:
            test_size (float): Proportion of dataset to use for testing (default 0.2 for 80:20 split)
            random_state (int): Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # Model components
        self.sentiment_scorer = SentimentScorer(max_features=5000)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        self.logistic_model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0
        )
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
        # Results storage
        self.model_fitted = False
        self.predictions = None
        self.classification_metrics = None
    
    def create_binary_labels(self, ratings):
        """
        Convert 1-5 ratings to binary classification
        1-3: Negative (0)
        4-5: Positive (1)
        """
        return (ratings >= 4).astype(int)
    
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the hotel reviews data
        
        Args:
            data_path (str): Path to the preprocessed CSV file
            
        Returns:
            pd.DataFrame: Processed dataframe ready for modeling
        """
        print(f"\n{'='*60}")
        print("LOADING AND PREPROCESSING DATA")
        print(f"{'='*60}")
        
        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} reviews from {data_path}")
        
        # Check required columns
        required_cols = ['Processed_Review', 'Rating']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing values
        initial_len = len(df)
        df = df[required_cols].dropna()
        removed = initial_len - len(df)
        if removed > 0:
            print(f"⚠️  Removed {removed} rows with missing values")
        
        # Additional preprocessing on the processed reviews
        print("🔄 Applying additional text preprocessing...")
        df['Final_Processed_Review'] = df['Processed_Review'].apply(
            lambda x: preprocess_text(x, remove_punct=True, remove_stops=True, keep_negations=True)
        )
        
        # Remove empty reviews after final preprocessing
        df = df[df['Final_Processed_Review'].str.strip() != '']
        print(f"✓ Final dataset size: {len(df)} reviews")
        
        # Create binary labels
        df['Binary_Rating'] = self.create_binary_labels(df['Rating'])
        
        # Display distribution
        print(f"\nRating Distribution:")
        for rating in sorted(df['Rating'].unique()):
            count = (df['Rating'] == rating).sum()
            percentage = (count / len(df)) * 100
            print(f"  Rating {rating}: {count} ({percentage:.1f}%)")
        
        print(f"\nBinary Classification Distribution:")
        binary_counts = df['Binary_Rating'].value_counts()
        print(f"  Negative (0): {binary_counts.get(0, 0)} ({(binary_counts.get(0, 0)/len(df)*100):.1f}%)")
        print(f"  Positive (1): {binary_counts.get(1, 0)} ({(binary_counts.get(1, 0)/len(df)*100):.1f}%)")
        
        return df
    
    def assign_sentiment_scores(self, df):
        """
        Assign sentiment scores to each word using TF-IDF based scoring
        
        Args:
            df (pd.DataFrame): DataFrame with processed reviews
            
        Returns:
            pd.DataFrame: DataFrame with additional sentiment features
        """
        print(f"\n{'='*60}")
        print("ASSIGNING SENTIMENT SCORES")
        print(f"{'='*60}")
        
        # Fit sentiment scorer
        self.sentiment_scorer.fit(
            df['Final_Processed_Review'].values, 
            df['Rating'].values
        )
        
        # Calculate sentiment scores for each review
        sentiment_scores = []
        word_counts = []
        
        for text in df['Final_Processed_Review']:
            result = self.sentiment_scorer.score_text(text)
            sentiment_scores.append(result['overall_score'])
            word_counts.append(result['scored_words'])
        
        df['Sentiment_Score'] = sentiment_scores
        df['Scored_Words_Count'] = word_counts
        
        print(f"✓ Assigned sentiment scores to {len(df)} reviews")
        print(f"Average sentiment score: {np.mean(sentiment_scores):.2f}")
        print(f"Average words scored per review: {np.mean(word_counts):.1f}")
        
        return df
    
    def split_data(self, df):
        """
        Split data into training and testing sets (80:20)
        
        Args:
            df (pd.DataFrame): Processed dataframe
        """
        print(f"\n{'='*60}")
        print("SPLITTING DATA (80:20 TRAIN:TEST)")
        print(f"{'='*60}")
        
        X = df['Final_Processed_Review']
        y = df['Binary_Rating']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set: {len(self.X_train)} samples ({(1-self.test_size)*100:.0f}%)")
        print(f"✓ Testing set: {len(self.X_test)} samples ({self.test_size*100:.0f}%)")
        
        # Check class distribution in splits
        train_pos = self.y_train.sum()
        test_pos = self.y_test.sum()
        print(f"\nClass distribution:")
        print(f"  Training - Positive: {train_pos}, Negative: {len(self.y_train)-train_pos}")
        print(f"  Testing - Positive: {test_pos}, Negative: {len(self.y_test)-test_pos}")
    
    def vectorize_text(self):
        """Convert text to TF-IDF vectors"""
        print(f"\n{'='*60}")
        print("VECTORIZING TEXT WITH TF-IDF")
        print(f"{'='*60}")
        
        # Fit and transform training data
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        # Transform test data
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)
        
        print(f"✓ TF-IDF vocabulary size: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        print(f"✓ Training matrix shape: {self.X_train_tfidf.shape}")
        print(f"✓ Testing matrix shape: {self.X_test_tfidf.shape}")
    
    def train_model(self):
        """Train the logistic regression model"""
        print(f"\n{'='*60}")
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print(f"{'='*60}")
        
        # Train the model
        self.logistic_model.fit(self.X_train_tfidf, self.y_train)
        
        # Get training accuracy
        train_predictions = self.logistic_model.predict(self.X_train_tfidf)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        
        print(f"✓ Model training completed")
        print(f"✓ Training accuracy: {train_accuracy:.4f}")
        
        self.model_fitted = True
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        if not self.model_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        
        # Make predictions
        self.predictions = self.logistic_model.predict(self.X_test_tfidf)
        prediction_proba = self.logistic_model.predict_proba(self.X_test_tfidf)
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, self.predictions)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.predictions, 
                                  target_names=['Negative', 'Positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    Negative  Positive")
        print(f"Negative    {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Positive    {cm[1,0]:6d}    {cm[1,1]:6d}")
        
        # Store results
        self.classification_metrics = {
            'accuracy': test_accuracy,
            'confusion_matrix': cm,
            'predictions': self.predictions,
            'probabilities': prediction_proba
        }
        
        return self.classification_metrics
    
    def visualize_results(self, output_dir='visualizations'):
        """Create visualizations of model results"""
        if not self.model_fitted:
            raise ValueError("Model must be trained before visualization")
        
        script_dir = Path(__file__).parent.resolve()
        viz_dir = script_dir / output_dir
        viz_dir.mkdir(exist_ok=True)
        
        sns.set_style('whitegrid')
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = self.classification_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - Logistic Regression', fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction Probability Distribution
        plt.figure(figsize=(12, 5))
        probabilities = self.classification_metrics['probabilities']
        
        plt.subplot(1, 2, 1)
        plt.hist(probabilities[:, 1], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Positive Class Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Positive Class Probabilities')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Separate by actual class
        pos_probs = probabilities[self.y_test == 1, 1]
        neg_probs = probabilities[self.y_test == 0, 1]
        
        plt.hist(neg_probs, bins=20, alpha=0.7, label='Actually Negative', color='red')
        plt.hist(pos_probs, bins=20, alpha=0.7, label='Actually Positive', color='green')
        plt.xlabel('Positive Class Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution by True Class')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {viz_dir}")
        print("  - confusion_matrix.png")
        print("  - probability_distributions.png")
    
    def get_feature_importance(self, top_n=20):
        """Get most important features (words) for classification"""
        if not self.model_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature names and coefficients
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        coefficients = self.logistic_model.coef_[0]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
        
        print(f"\n{'='*60}")
        print("TOP PREDICTIVE FEATURES")
        print(f"{'='*60}")
        print(f"{'Feature':<20} {'Coefficient':<12} {'Impact'}")
        print("-" * 50)
        
        for _, row in feature_importance.head(top_n).iterrows():
            impact = "Positive" if row['coefficient'] > 0 else "Negative"
            print(f"{row['feature']:<20} {row['coefficient']:<12.4f} {impact}")
        
        return feature_importance.head(top_n)
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text to classify
            
        Returns:
            dict: Prediction results with probability and classification
        """
        if not self.model_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the text
        processed_text = preprocess_text(text, remove_punct=True, remove_stops=True, keep_negations=True)
        
        # Vectorize
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.logistic_model.predict(text_tfidf)[0]
        probability = self.logistic_model.predict_proba(text_tfidf)[0]
        
        # Get sentiment score
        sentiment_result = self.sentiment_scorer.score_text(processed_text)
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'prediction_label': 'Positive' if prediction == 1 else 'Negative',
            'positive_probability': probability[1],
            'negative_probability': probability[0],
            'sentiment_score': sentiment_result['overall_score'],
            'sentiment_label': sentiment_result['overall_sentiment']
        }
    
    def save_model(self, filepath):
        """
        Save the trained model to a file using pickle
        
        Args:
            filepath (str): Path where to save the model (should end with .pkl)
        """
        if not self.model_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'sentiment_scorer': self.sentiment_scorer,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'logistic_model': self.logistic_model,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'model_fitted': self.model_fitted,
            'classification_metrics': self.classification_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file using pickle
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            HotelReviewNLPModel: Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls(
            test_size=model_data['test_size'],
            random_state=model_data['random_state']
        )
        
        # Restore saved components
        model.sentiment_scorer = model_data['sentiment_scorer']
        model.tfidf_vectorizer = model_data['tfidf_vectorizer']
        model.logistic_model = model_data['logistic_model']
        model.model_fitted = model_data['model_fitted']
        model.classification_metrics = model_data.get('classification_metrics')
        
        print(f"✓ Model loaded from: {filepath}")
        return model


def main():
    """Main function to run the complete NLP pipeline"""
    print("\n" + "="*80)
    print("HOTEL REVIEW NLP MODEL - COMPLETE PIPELINE")
    print("="*80)
    
    # Initialize model
    model = HotelReviewNLPModel(test_size=0.2, random_state=42)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load and preprocess data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'tripadvisor_hotel_reviews_preprocessed.csv')
    
    try:
        df = model.load_and_preprocess_data(data_path)
        
        # Assign sentiment scores
        df = model.assign_sentiment_scores(df)
        
        # Split data
        model.split_data(df)
        
        # Vectorize text
        model.vectorize_text()
        
        # Train model
        model.train_model()
        
        # Evaluate model
        metrics = model.evaluate_model()
        
        # Get feature importance
        important_features = model.get_feature_importance(top_n=15)
        
        # Create visualizations
        model.visualize_results()
        
        # Test with example reviews
        print(f"\n{'='*60}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'='*60}")
        
        example_reviews = [
            "This hotel was absolutely fantastic! The staff was incredible and the room was beautiful.",
            "Terrible experience. The room was dirty and the service was awful.",
            "It was okay, nothing special but not bad either.",
            "Amazing location, great food, but the room could use some updates.",
            "Worst hotel ever! Completely disgusted with everything!"
        ]
        
        for i, review in enumerate(example_reviews, 1):
            result = model.predict_sentiment(review)
            print(f"\nExample {i}:")
            print(f"Text: {result['original_text']}")
            print(f"Prediction: {result['prediction_label']} (Confidence: {result['positive_probability']:.2f})")
            print(f"Sentiment Score: {result['sentiment_score']:.2f} ({result['sentiment_label']})")
        
        # Save the trained model
        model_save_path = os.path.join(base_dir, 'trained_nlp_model.pkl')
        model.save_model(model_save_path)
        
        print(f"\n{'='*80}")
        print("NLP PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final Model Accuracy: {metrics['accuracy']:.4f}")
        print(f"Model saved to: {model_save_path}")
        print("="*80)
        
        return model
        
    except FileNotFoundError:
        print(f"\n❌ Error: Preprocessed data file not found at {data_path}")
        print("Please run data_preprocessor.py first to generate the required data.")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Set up the base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Run the main pipeline
    nlp_model = main()
