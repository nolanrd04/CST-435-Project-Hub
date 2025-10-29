"""
Sentiment Scorer for Hotel Reviews
Uses TF-IDF vectorization to assign sentiment scores to words on a 1-5 scale
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SentimentScorer:
    """
    A class to score sentiment of words using TF-IDF vectorization
    Assigns sentiment scores on a 1-5 scale based on word associations with ratings
    """
    
    def __init__(self, max_features=5000):
        """
        Initialize the sentiment scorer
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorization
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        self.word_sentiment_scores = {}
        self.sentiment_labels = {
            1: 'Very Negative',
            2: 'Negative',
            3: 'Neutral',
            4: 'Positive',
            5: 'Very Positive'
        }
        
    def fit(self, texts, ratings):
        """
        Fit the TF-IDF vectorizer and calculate sentiment scores for each word
        
        Args:
            texts: Array of review texts
            ratings: Array of ratings (1-5) corresponding to each text
        """
        print(f"\n{'='*50}")
        print(f"TF-IDF Vectorization and Sentiment Scoring")
        print(f"{'='*50}")
        
        # Fit and transform the texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Vocabulary size: {len(feature_names)}")
        print(f"Total documents: {len(texts)}")
        
        # Calculate sentiment score for each word
        # For each word, we'll compute the average rating of documents where it appears
        # weighted by the TF-IDF score
        
        for i, word in enumerate(feature_names):
            # Get TF-IDF scores for this word across all documents
            word_scores = tfidf_matrix[:, i].toarray().flatten()
            
            # Calculate weighted average rating
            # Only consider documents where the word appears (score > 0)
            mask = word_scores > 0
            if mask.sum() > 0:
                weighted_rating = np.average(ratings[mask], weights=word_scores[mask])
                self.word_sentiment_scores[word] = {
                    'score': weighted_rating,
                    'sentiment': self.map_rating_to_sentiment(round(weighted_rating)),
                    'frequency': mask.sum()
                }
        
        print(f"Sentiment scores calculated for {len(self.word_sentiment_scores)} words")
        
    def map_rating_to_sentiment(self, rating):
        """
        Map numeric rating (1-5) to sentiment label
        
        Args:
            rating (int): Rating from 1-5
            
        Returns:
            str: Sentiment label
        """
        return self.sentiment_labels.get(rating, 'Unknown')
    
    def get_word_score(self, word):
        """
        Get sentiment score for a specific word
        
        Args:
            word (str): Word to get score for
            
        Returns:
            dict: Dictionary with score, sentiment label, and frequency
        """
        return self.word_sentiment_scores.get(word.lower(), None)
    
    def get_top_words_by_sentiment(self, sentiment_rating, n=20):
        """
        Get top words for a specific sentiment rating
        
        Args:
            sentiment_rating (int): Rating (1-5) to get top words for
            n (int): Number of top words to return
            
        Returns:
            list: List of tuples (word, score_info)
        """
        # Filter words by sentiment category
        filtered_words = [
            (word, info) for word, info in self.word_sentiment_scores.items()
            if round(info['score']) == sentiment_rating
        ]
        
        # Sort by frequency (how often the word appears)
        sorted_words = sorted(filtered_words, key=lambda x: x[1]['frequency'], reverse=True)
        
        return sorted_words[:n]
    
    def score_text(self, text):
        """
        Calculate sentiment score for a given text
        
        Args:
            text (str): Text to score
            
        Returns:
            dict: Dictionary with overall score, sentiment label, and word scores
        """
        # Tokenize the text (simple split for now)
        words = text.split()
        
        word_scores = []
        for word in words:
            score_info = self.get_word_score(word)
            if score_info:
                word_scores.append({
                    'word': word,
                    'score': score_info['score'],
                    'sentiment': score_info['sentiment']
                })
        
        if word_scores:
            avg_score = np.mean([w['score'] for w in word_scores])
            return {
                'overall_score': avg_score,
                'overall_sentiment': self.map_rating_to_sentiment(round(avg_score)),
                'word_scores': word_scores,
                'scored_words': len(word_scores),
                'total_words': len(words)
            }
        else:
            return {
                'overall_score': 3.0,  # Neutral default
                'overall_sentiment': 'Neutral',
                'word_scores': [],
                'scored_words': 0,
                'total_words': len(words)
            }
    
    def count_sentiments(self, df, text_column='Processed_Review'):
        """
        Count positive, negative, and neutral items in the dataset
        
        Args:
            df (pd.DataFrame): DataFrame with reviews
            text_column (str): Column containing text
            
        Returns:
            dict: Counts of positive, negative, neutral reviews
        """
        sentiment_counts = {
            'positive': 0,      # Ratings 4-5
            'negative': 0,      # Ratings 1-2
            'neutral': 0        # Rating 3
        }
        
        for text in df[text_column]:
            result = self.score_text(text)
            score = result['overall_score']
            
            if score >= 4:
                sentiment_counts['positive'] += 1
            elif score <= 2:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        return sentiment_counts
    
    def visualize_sentiment_distribution(self, output_dir='visualizations'):
        """
        Create visualizations showing the distribution of sentiment scores
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        script_dir = Path(__file__).parent.resolve()
        viz_dir = script_dir / output_dir
        viz_dir.mkdir(exist_ok=True)
        
        sns.set_style('whitegrid')
        
        # 1. Distribution of word sentiment scores
        plt.figure(figsize=(12, 6))
        scores = [info['score'] for info in self.word_sentiment_scores.values()]
        plt.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Sentiment Score (1-5)', fontsize=12)
        plt.ylabel('Number of Words', fontsize=12)
        plt.title('Distribution of Word Sentiment Scores', fontsize=14, fontweight='bold')
        plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Neutral (3.0)')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'word_sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Count of words in each sentiment category
        plt.figure(figsize=(10, 6))
        sentiment_counts = {}
        for info in self.word_sentiment_scores.values():
            sentiment = info['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        sentiments = [self.sentiment_labels[i] for i in sorted(self.sentiment_labels.keys())]
        counts = [sentiment_counts.get(s, 0) for s in sentiments]
        colors = ['#d62728', '#ff7f0e', '#ffff00', '#2ca02c', '#1f77b4']
        
        plt.bar(sentiments, counts, color=colors, edgecolor='black')
        plt.xlabel('Sentiment Category', fontsize=12)
        plt.ylabel('Number of Words', fontsize=12)
        plt.title('Words by Sentiment Category', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'sentiment_category_counts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {viz_dir}")
        print("  - word_sentiment_distribution.png")
        print("  - sentiment_category_counts.png")


def main():
    """
    Main function to run sentiment scoring
    """
    print("\n" + "="*50)
    print("SENTIMENT SCORER FOR HOTEL REVIEWS")
    print("="*50)
    
    # Get current script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Load preprocessed data
    data_path = script_dir / 'tripadvisor_hotel_reviews_preprocessed.csv'
    
    if not data_path.exists():
        print(f"\nError: Preprocessed data not found at {data_path}")
        print("Please run data_preprocessor.py first to generate the preprocessed data.")
        return
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Remove rows with missing values
    df_clean = df[['Processed_Review', 'Rating']].dropna()
    
    print(f"Total reviews: {len(df_clean)}")
    print(f"\nRating Distribution:")
    for rating in sorted(df_clean['Rating'].unique()):
        count = (df_clean['Rating'] == rating).sum()
        percentage = (count / len(df_clean)) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")
    
    # Initialize sentiment scorer
    scorer = SentimentScorer(max_features=5000)
    
    # Fit the scorer
    scorer.fit(df_clean['Processed_Review'].values, df_clean['Rating'].values)
    
    # Show top words for each sentiment category
    print(f"\n{'='*50}")
    print("Top Words for Each Sentiment Category")
    print(f"{'='*50}")
    
    for rating in sorted(scorer.sentiment_labels.keys()):
        sentiment = scorer.map_rating_to_sentiment(rating)
        print(f"\n{rating} - {sentiment}:")
        top_words = scorer.get_top_words_by_sentiment(rating, n=15)
        for word, info in top_words:
            print(f"  {word:20s} | Score: {info['score']:.2f} | Frequency: {info['frequency']}")
    
    # Count sentiments in dataset
    print(f"\n{'='*50}")
    print("Sentiment Counts in Dataset")
    print(f"{'='*50}")
    sentiment_counts = scorer.count_sentiments(df_clean)
    print(f"Positive (score >= 4): {sentiment_counts['positive']}")
    print(f"Neutral (2 < score < 4): {sentiment_counts['neutral']}")
    print(f"Negative (score <= 2): {sentiment_counts['negative']}")
    
    # Create visualizations
    scorer.visualize_sentiment_distribution()
    
    # Example: Score some sample reviews
    print(f"\n{'='*50}")
    print("Example Sentiment Scoring")
    print(f"{'='*50}")
    
    example_reviews = [
        "This hotel was absolutely amazing! The staff was friendly and the room was beautiful.",
        "Terrible experience. The room was dirty and the service was horrible.",
        "It was okay. Nothing special but nothing terrible either.",
    ]
    
    for i, review in enumerate(example_reviews, 1):
        result = scorer.score_text(review)
        print(f"\nReview {i}: {review}")
        print(f"Overall Score: {result['overall_score']:.2f} ({result['overall_sentiment']})")
        print(f"Words scored: {result['scored_words']}/{result['total_words']}")
        print("Word-level scores:")
        for word_info in result['word_scores'][:5]:  # Show first 5 words
            print(f"  {word_info['word']:15s} → {word_info['score']:.2f} ({word_info['sentiment']})")
    
    print(f"\n{'='*50}")
    print("Sentiment Scoring Complete!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
