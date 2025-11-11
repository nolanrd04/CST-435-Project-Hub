# THIS FILE SHOULD NOT RUN ON THE FRONT END. DO NOT HAVE ANY API CALLS TO IT.
# THIS FILE IS SOLELY FOR GENERATING A USEABLE DATASET.
# RUNNING THIS FILE OR ITS FUNCTIONS ON THE FRONT END COULD CAUSE COMPUTATIONAL ERRORS WITH THE LIMITED HARDWARE RESOURCES WE HAVE ON RENDER.

import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
import json
import re
import string

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, "data")
os.makedirs(DATA_PATH, exist_ok=True)

# Check if JSON file already exists
target_file = "900k Definitive Spotify Dataset.json"
dst_file = os.path.join(DATA_PATH, target_file)

if os.path.exists(dst_file):
    print(f"✓ Found existing dataset: {target_file}")
    print(f"  File size: {os.path.getsize(dst_file) / (1024**2):.2f} MB")
    print("  Skipping download...\n")
else:
    # Download dataset (kagglehub caches it in a random location)
    cached_path = kagglehub.dataset_download("devdope/900k-spotify")
    print(f"Dataset cached at: {cached_path}")

    # Copy only the definitive dataset JSON file
    src_file = os.path.join(cached_path, target_file)

    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"\n✓ Copied: {target_file}")
        print(f"File size: {os.path.getsize(dst_file) / (1024**2):.2f} MB")
    else:
        print(f"\n✗ Error: {target_file} not found in dataset")
        print("Available files:")
        for file in os.listdir(cached_path):
            print(f"  - {file}")


# Convert json to txt for RNN training
def preprocess_lyrics(text):
    """
    Preprocess lyrics for RNN training.
    
    Steps:
    1. Convert to lowercase (for consistency in lyric generation)
    2. Remove URLs
    3. Remove extra whitespace and newlines
    4. Remove punctuation EXCEPT apostrophes (for contractions like "don't", "it's")
    5. Keep line breaks to preserve song structure (important for lyrics)
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Normalize whitespace but preserve line breaks
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single
    
    # Remove punctuation except apostrophes (keep contractions)
    punct_to_remove = string.punctuation.replace("'", "")
    for char in punct_to_remove:
        text = text.replace(char, "")
    
    # Clean up any remaining whitespace issues
    text = text.strip()
    
    return text


# Extract and preprocess text from JSON
print("\n" + "="*50)
print("EXTRACTING AND PREPROCESSING LYRICS")
print("="*50)

try:
    print(f"\nOpening file: {dst_file}")
    print(f"File exists: {os.path.exists(dst_file)}")
    print(f"File size: {os.path.getsize(dst_file) / (1024**2):.2f} MB")
    
    with open(dst_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nLoaded {len(data)} records from JSON")
    
    # Extract lyrics field (assuming it's called 'text' or similar)
    all_lyrics = []
    current_size_bytes = 0
    max_size_bytes = 20 * 1024 * 1024  # 20 MB limit
    
    for i, record in enumerate(data):
        # Try different possible field names for lyrics
        lyrics = record.get('lyrics') or record.get('text') or record.get('lyric') or ""
        if lyrics:
            preprocessed = preprocess_lyrics(lyrics)
            if preprocessed:
                # Check if adding this lyric would exceed the size limit
                lyric_size = len(preprocessed.encode('utf-8')) + 1  # +1 for newline
                if current_size_bytes + lyric_size > max_size_bytes:
                    print(f"\n⚠ Reached 20 MB limit at {i} records")
                    break
                
                all_lyrics.append(preprocessed)
                current_size_bytes += lyric_size
        
        if (i + 1) % 100000 == 0:
            size_mb = current_size_bytes / (1024**2)
            print(f"  Processed {i + 1} records... ({size_mb:.2f} MB)")
    
    print(f"\nSuccessfully extracted {len(all_lyrics)} lyric entries")

    print("Enter a txt file to save to, or leave blank to use default (lyrics_preprocessed.txt): ", end="")
    user_input = input().strip()
    if user_input:
        output_file = user_input
    else:
        output_file = os.path.join(DATA_PATH, "lyrics_preprocessed.txt")
    
    # Join all lyrics with a newline separator
    combined_text = "\n".join(all_lyrics)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    num_lines = len(all_lyrics)
    num_chars = len(combined_text)
    
    print(f"\n✓ Saved preprocessed lyrics to: lyrics_preprocessed.txt")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Number of songs/entries: {num_lines}")
    print(f"  - Total characters: {num_chars:,}")
    
    # Show a sample
    print(f"\n--- SAMPLE (first 500 characters) ---")
    print(combined_text[:500])
    print("..." if len(combined_text) > 500 else "")

except json.JSONDecodeError as e:
    print(f"\n✗ Error: Could not decode JSON file")
    print(f"  JSON Error: {str(e)}")
    print(f"  Line: {e.lineno}, Column: {e.colno}")
    print("\n  This might mean:")
    print("  - The file is incomplete or corrupted")
    print("  - The file is in NDJSON format (newline-delimited JSON) instead of standard JSON")
    print("\n  Trying alternative parsing method (NDJSON)...")

    print("Enter desired txt file size in mb (max limit 1000mb) (default 20mb): ", end="")
    size_input = input().strip()
    try:
        size_mb = int(size_input)
        if size_mb <= 0 or size_mb > 1000:
            size_mb = 20
    except ValueError:
        size_mb = 20
    
    try:
        all_lyrics = []
        current_size_bytes = 0
        max_size_bytes = size_mb * 1024 * 1024  # Convert MB to bytes
        
        with open(dst_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    lyrics = record.get('text', "")
                    if lyrics:
                        preprocessed = preprocess_lyrics(lyrics)
                        if preprocessed:
                            # Check if adding this lyric would exceed the size limit
                            lyric_size = len(preprocessed.encode('utf-8')) + 1  # +1 for newline
                            if current_size_bytes + lyric_size > max_size_bytes:
                                print(f"\n⚠ Reached {max_size_bytes / (1024**2):.2f} MB limit at {line_num} records")
                                break
                            
                            all_lyrics.append(preprocessed)
                            current_size_bytes += lyric_size
                    
                    if line_num % 10000 == 0:
                        size_mb = current_size_bytes / (1024**2)
                        print(f"  Processed {line_num} records... ({size_mb:.2f} MB / {max_size_bytes / (1024**2):.2f} MB)")
                except json.JSONDecodeError:
                    continue
        
        if all_lyrics:
            print(f"\n✓ Successfully parsed as NDJSON format!")
            print(f"  Extracted {len(all_lyrics)} lyric entries")

            print("Enter a txt file to save to, or leave blank to use default (lyrics_preprocessed.txt): ", end="")
            user_input = input().strip()
            if user_input:
                output_file = os.path.join(DATA_PATH, user_input)
            else:
                output_file = os.path.join(DATA_PATH, "lyrics_preprocessed.txt")

            combined_text = "\n".join(all_lyrics)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            file_size_mb = os.path.getsize(output_file) / (1024**2)
            num_chars = len(combined_text)
            
            print(f"\n✓ Saved preprocessed lyrics to: {output_file}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            print(f"  - Number of songs/entries: {len(all_lyrics)}")
            print(f"  - Total characters: {num_chars:,}")
            
            # Show a sample
            print(f"\n--- SAMPLE (first 500 characters) ---")
            print(combined_text[:500])
            print("..." if len(combined_text) > 500 else "")
        else:
            print("Could not extract any lyrics even with NDJSON parsing")
    except Exception as e2:
        print(f"NDJSON parsing also failed: {str(e2)}")

except FileNotFoundError:
    print(f"\n✗ Error: File not found at {dst_file}")
except Exception as e:
    print(f"✗ Unexpected error: {str(e)}")
