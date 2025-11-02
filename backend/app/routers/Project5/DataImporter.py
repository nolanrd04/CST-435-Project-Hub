# THIS FILE SHOULD NOT RUN ON THE FRONT END. DO NOT HAVE ANY API CALLS TO IT.
# THIS FILE IS SOLELY FOR DOWNLOADING AND IMPORTING DATASET.
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


def preprocess_lyrics(text):
    """
    Preprocess lyrics for RNN training.
    
    Steps:
    1. Convert to lowercase (for consistency in lyric generation)
    2. Remove URLs
    3. Replace punctuation EXCEPT apostrophes with spaces (to avoid word concatenation)
    4. Normalize whitespace and preserve line breaks
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Replace punctuation except apostrophes with spaces (prevents word concatenation)
    punct_to_remove = string.punctuation.replace("'", "")
    for char in punct_to_remove:
        text = text.replace(char, " ")
    
    # Normalize whitespace but preserve line breaks
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single
    text = re.sub(r' *\n *', '\n', text)  # Trim spaces around newlines
    
    # Clean up any remaining whitespace issues
    text = text.strip()
    
    return text


def import_spotify_dataset():
    """
    Download and import the Spotify lyrics dataset.
    Returns the path to the preprocessed lyrics text file.
    """
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Check if preprocessed file already exists
    output_file = os.path.join(DATA_PATH, "lyrics_preprocessed.txt")
    if os.path.exists(output_file):
        print(f"✓ Found existing preprocessed lyrics: lyrics_preprocessed.txt")
        print(f"  File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
        return output_file
    
    # Check if JSON file already exists
    target_file = "900k Definitive Spotify Dataset.json"
    dst_file = os.path.join(DATA_PATH, target_file)

    if os.path.exists(dst_file):
        print(f"✓ Found existing dataset: {target_file}")
        print(f"  File size: {os.path.getsize(dst_file) / (1024**2):.2f} MB")
        print("  Skipping download...\n")
    else:
        print("Downloading Spotify dataset...")
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
            raise FileNotFoundError(f"Dataset file {target_file} not found")

    # Extract and preprocess text from JSON
    print("\n" + "="*50)
    print("EXTRACTING AND PREPROCESSING LYRICS")
    print("="*50)

    try:
        print(f"\nOpening file: {dst_file}")
        print(f"File exists: {os.path.exists(dst_file)}")
        print(f"File size: {os.path.getsize(dst_file) / (1024**2):.2f} MB")
        print("Processing file in streaming mode to avoid memory issues...")
        
        # Try streaming JSON parsing first (assumes JSON array format)
        all_lyrics = []
        current_size_bytes = 0
        max_size_bytes = 80 * 1024 * 1024  # 80 MB limit (git-friendly, under 100MB)
        size_limit_reached = False
        
        try:
            import ijson  # Try to use ijson for streaming if available
            print("Using ijson for memory-efficient streaming...")
            print("Starting with concatenated JSON objects format (known to work)...")
            
            # Start directly with concatenated JSON objects format since we know it works
            with open(dst_file, 'r', encoding='utf-8') as f:
                buffer = ""
                records_processed = 0
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Add line to buffer
                    if buffer:
                        buffer += " " + line
                    else:
                        buffer = line
                    
                    # Try to parse complete JSON objects from buffer
                    while buffer.strip():
                        try:
                            # Find the end of a JSON object
                            decoder = json.JSONDecoder()
                            obj, idx = decoder.raw_decode(buffer)
                            
                            # Process the found object - extract lyrics from 'text' field
                            lyrics = obj.get('text', "")
                            if lyrics and lyrics.strip():
                                preprocessed = preprocess_lyrics(lyrics)
                                if preprocessed:
                                    lyric_size = len(preprocessed.encode('utf-8')) + 1
                                    if current_size_bytes + lyric_size > max_size_bytes:
                                        print(f"\n⚠ Reached 80 MB limit at {records_processed} records")
                                        # Set flag to break out of all loops
                                        size_limit_reached = True
                                        break
                                    
                                    all_lyrics.append(preprocessed)
                                    current_size_bytes += lyric_size
                            
                            records_processed += 1
                            if records_processed % 5000 == 0:  # More frequent updates
                                size_mb = current_size_bytes / (1024**2)
                                print(f"  Processed {records_processed:,} records... ({size_mb:.2f} MB / 80 MB)")
                            
                            # Check if we hit size limit
                            if current_size_bytes >= max_size_bytes:
                                size_limit_reached = True
                                break
                            
                            # Remove processed JSON from buffer and continue
                            buffer = buffer[idx:].lstrip()
                            
                        except json.JSONDecodeError:
                            # If we can't parse, we need more data
                            break
                    
                    # Break out of outer loop if size limit reached
                    if size_limit_reached:
                        break
        
        except (ImportError, NameError):
            # Fallback: Process line by line assuming it might be NDJSON or similar
            print("ijson not available or failed, trying manual line-by-line processing...")
            
            size_limit_reached = False  # Initialize flag for fallback method
            with open(dst_file, 'r', encoding='utf-8') as f:
                # Skip opening bracket if it exists
                first_char = f.read(1)
                if first_char != '[':
                    f.seek(0)  # Reset if not array format
                
                records_processed = 0
                buffer = ""
                brace_count = 0
                in_string = False
                escape_next = False
                
                while True:
                    chunk = f.read(8192)  # Read 8KB chunks
                    if not chunk or size_limit_reached:
                        break
                    
                    for char in chunk:
                        if escape_next:
                            escape_next = False
                            buffer += char
                            continue
                            
                        if char == '\\' and in_string:
                            escape_next = True
                            buffer += char
                            continue
                            
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                
                        buffer += char
                        
                        # When we have a complete JSON object
                        if not in_string and brace_count == 0 and buffer.strip().endswith('}'):
                            try:
                                # Clean up the buffer
                                json_str = buffer.strip().rstrip(',').strip()
                                if json_str:
                                    record = json.loads(json_str)
                                    
                                    # Extract lyrics from the 'text' field (Spotify dataset format)
                                    lyrics = record.get('text', "")
                                    if lyrics:
                                        preprocessed = preprocess_lyrics(lyrics)
                                        if preprocessed:
                                            # Check if adding this lyric would exceed the size limit
                                            lyric_size = len(preprocessed.encode('utf-8')) + 1
                                            if current_size_bytes + lyric_size > max_size_bytes:
                                                print(f"\n⚠ Reached 80 MB limit at {records_processed} records")
                                                size_limit_reached = True
                                                break
                                            
                                            all_lyrics.append(preprocessed)
                                            current_size_bytes += lyric_size
                                    
                                    records_processed += 1
                                    if records_processed % 10000 == 0:
                                        size_mb = current_size_bytes / (1024**2)
                                        print(f"  Processed {records_processed:,} records... ({size_mb:.2f} MB / 80 MB)")
                            
                            except json.JSONDecodeError:
                                if json_str and records_processed > 0:
                                    break
                            
                            buffer = ""
                            
                            # Check if size limit reached
                            if size_limit_reached:
                                break
        
        print(f"\nSuccessfully extracted {len(all_lyrics)} lyric entries")
        
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
        
        return output_file

    except json.JSONDecodeError as e:
        print(f"\n✗ Error: Could not decode JSON file")
        print(f"  JSON Error: {str(e)}")
        print(f"  Line: {e.lineno}, Column: {e.colno}")
        print("\n  This might mean:")
        print("  - The file is incomplete or corrupted")
        print("  - The file is in NDJSON format (newline-delimited JSON) instead of standard JSON")
        print("\n  Trying alternative parsing method (NDJSON)...")
        
        try:
            print("Trying NDJSON line-by-line processing (memory efficient)...")
            all_lyrics = []
            current_size_bytes = 0
            max_size_bytes = 80 * 1024 * 1024  # 80 MB limit (git-friendly, under 100MB)
            
            with open(dst_file, 'r', encoding='utf-8') as f:
                line_num = 0
                valid_records = 0
                
                for line in f:
                    line_num += 1
                    line = line.strip()
                    
                    # Skip empty lines and array brackets
                    if not line or line in ['[', ']', ',']:
                        continue
                    
                    # Remove trailing comma if present
                    if line.endswith(','):
                        line = line[:-1]
                    
                    try:
                        record = json.loads(line)
                        lyrics = record.get('text', "")
                        if lyrics:
                            preprocessed = preprocess_lyrics(lyrics)
                            if preprocessed:
                                # Check if adding this lyric would exceed the size limit
                                lyric_size = len(preprocessed.encode('utf-8')) + 1  # +1 for newline
                                if current_size_bytes + lyric_size > max_size_bytes:
                                    print(f"\n⚠ Reached 80 MB limit at {valid_records} valid records (line {line_num})")
                                    break
                                
                                all_lyrics.append(preprocessed)
                                current_size_bytes += lyric_size
                                valid_records += 1
                        
                        if line_num % 50000 == 0:  # More frequent updates for large files
                            size_mb = current_size_bytes / (1024**2)
                            print(f"  Processed {line_num:,} lines, {valid_records:,} valid records... ({size_mb:.2f} MB / 80 MB)")
                    
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue
                    
                    # Memory management: force garbage collection every 100k records
                    if line_num % 100000 == 0:
                        import gc
                        gc.collect()
            
            if all_lyrics:
                print(f"\n✓ Successfully parsed as NDJSON format!")
                print(f"  Extracted {len(all_lyrics)} lyric entries")
                
                combined_text = "\n".join(all_lyrics)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                
                file_size_mb = os.path.getsize(output_file) / (1024**2)
                num_chars = len(combined_text)
                
                print(f"\n✓ Saved preprocessed lyrics to: lyrics_preprocessed.txt")
                print(f"  - File size: {file_size_mb:.2f} MB")
                print(f"  - Number of songs/entries: {len(all_lyrics)}")
                print(f"  - Total characters: {num_chars:,}")
                
                # Show a sample
                print(f"\n--- SAMPLE (first 500 characters) ---")
                print(combined_text[:500])
                print("..." if len(combined_text) > 500 else "")
                
                return output_file
            else:
                raise ValueError("Could not extract any lyrics even with NDJSON parsing")
        except Exception as e2:
            raise RuntimeError(f"NDJSON parsing also failed: {str(e2)}")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {dst_file}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    try:
        output_path = import_spotify_dataset()
        print(f"\n✓ Dataset successfully imported and preprocessed at: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to import dataset: {str(e)}")