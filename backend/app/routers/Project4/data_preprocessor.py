# Install dependencies as needed:
# pip install kagglehub[pandas-datasets] nltk pandas
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

########## DOWNLOAD DATASET ##########
    
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Download the dataset (kagglehub downloads to its cache directory)
downloaded_path = kagglehub.dataset_download("andrewmvd/trip-advisor-hotel-reviews")
print(f"Dataset downloaded to: {downloaded_path}")

# The dataset file we want
dataset_filename = "tripadvisor_hotel_reviews.csv"
source_file = os.path.join(downloaded_path, dataset_filename)

# Where we want to save it (in the same directory as this script)
destination_file = os.path.join(BASE_DIR, dataset_filename)

# Copy the file to our project directory
if os.path.exists(source_file):
    print(f"\nCopying dataset to project directory...")
    shutil.copy2(source_file, destination_file)
    print(f"Dataset copied to: {destination_file}")
else:
    print(f"\n! Warning: Expected file '{dataset_filename}' not found. !")
    print(f"   Available files in download directory:")
    for file in os.listdir(downloaded_path):
        print(f"   - {file}")
    
    # If the filename is different, you can adjust it above


########## TEXT PREPROCESSING FUNCTIONS ##########

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("\nDownloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download('punkt_tab', quiet=True)
    
    print("✓ NLTK resources ready")


def remove_punctuation(text):
    """
    Remove punctuation from text
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text without punctuation
    """
    if pd.isna(text):
        return ""
    
    # Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def remove_stopwords(text, keep_negations=True):
    """
    Remove stopwords (words that don't add sentimental value)
    
    Args:
        text (str): Input text
        keep_negations (bool): Keep negation words (not, no, never) for sentiment
    
    Returns:
        str: Text without stopwords
    """
    if pd.isna(text):
        return ""
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Keep negation words as they're important for sentiment
    if keep_negations:
        negations = {'not', 'no', 'nor', 'never', 'neither', 'nobody', 'nothing', 
                     'nowhere', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', 
                     "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
                     "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                     'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                     'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                     'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        stop_words = stop_words - negations
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)


def clean_text(text):
    """
    Comprehensive text cleaning:
    - Convert to lowercase
    - Remove URLs
    - Remove extra whitespace
    - Remove numbers (optional)
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_text(text, remove_punct=True, remove_stops=True, keep_negations=True):
    """
    Complete text preprocessing pipeline
    
    Args:
        text (str): Input text
        remove_punct (bool): Remove punctuation
        remove_stops (bool): Remove stopwords
        keep_negations (bool): Keep negation words for sentiment
    
    Returns:
        str: Preprocessed text
    """
    # Clean text
    text = clean_text(text)
    
    # Remove punctuation
    if remove_punct:
        text = remove_punctuation(text)
    
    # Remove stopwords
    if remove_stops:
        text = remove_stopwords(text, keep_negations=keep_negations)
    
    return text


def preprocess_dataframe(df, text_column='Review', keep_negations=True):
    """
    Preprocess all reviews in a dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the column containing text
        keep_negations (bool): Keep negation words for sentiment
    
    Returns:
        pd.DataFrame: Dataframe with preprocessed text
    """
    print(f"\n🔄 Preprocessing {len(df)} reviews...")
    
    # Create new column with preprocessed text
    df['Processed_Review'] = df[text_column].apply(
        lambda x: preprocess_text(x, keep_negations=keep_negations)
    )
    
    # Remove empty reviews after preprocessing
    original_count = len(df)
    df = df[df['Processed_Review'].str.strip() != '']
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} empty reviews after preprocessing")
    
    print(f"✓ Preprocessing complete! {len(df)} reviews ready")
    
    return df


########## MAIN EXECUTION ##########

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HOTEL REVIEWS - DATA PREPROCESSING")
    print("="*80)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Check if dataset exists
    if os.path.exists(destination_file):
        print(f"\n📂 Loading dataset from: {destination_file}")
        
        # Load the dataset
        df = pd.read_csv(destination_file)
        print(f"✓ Loaded {len(df)} reviews")
        print(f"\nColumns: {list(df.columns)}")
        
        # Show sample data
        print("\n" + "="*80)
        print("SAMPLE DATA (Before Preprocessing)")
        print("="*80)
        print(df.head(3))
        
        # Preprocess the data
        df = preprocess_dataframe(df, text_column='Review', keep_negations=True)
        
        # Show processed samples
        print("\n" + "="*80)
        print("SAMPLE DATA (After Preprocessing)")
        print("="*80)
        print("\nOriginal vs Processed:")
        for idx in range(min(3, len(df))):
            print(f"\n--- Review {idx + 1} ---")
            print(f"Original:  {df.iloc[idx]['Review'][:100]}...")
            print(f"Processed: {df.iloc[idx]['Processed_Review'][:100]}...")
        
        # Save preprocessed data
        processed_file = os.path.join(BASE_DIR, "tripadvisor_hotel_reviews_preprocessed.csv")
        df.to_csv(processed_file, index=False)
        print(f"\n✓ Preprocessed data saved to: {processed_file}")
        
        # Statistics
        print("\n" + "="*80)
        print("PREPROCESSING STATISTICS")
        print("="*80)
        
        avg_original_length = df['Review'].str.len().mean()
        avg_processed_length = df['Processed_Review'].str.len().mean()
        reduction = ((avg_original_length - avg_processed_length) / avg_original_length) * 100
        
        print(f"Average review length:")
        print(f"  Original:  {avg_original_length:.1f} characters")
        print(f"  Processed: {avg_processed_length:.1f} characters")
        print(f"  Reduction: {reduction:.1f}%")
        
    else:
        print(f"\n⚠️  Dataset not found at: {destination_file}")
        print("Please run the download section first.")


