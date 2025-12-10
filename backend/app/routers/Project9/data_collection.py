#!/usr/bin/env python3
"""
Enhanced Image Dataset Collector using Search Engines
Dynamically searches for fruit images using Google and Bing.
"""

import os
import time
import json
import logging
import argparse
import hashlib
import re
import random
from pathlib import Path
from typing import List, Set, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from io import BytesIO
from PIL import Image, ImageFilter
import imagehash
from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    np = None

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None
    print("Warning: BeautifulSoup4 not found. Install with: pip install beautifulsoup4")
    print("Search functionality will be limited without BeautifulSoup4")

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
DEFAULT_FRUITS = ["orange", "strawberry", "blackberry", "pineapple", "banana"]
DEFAULT_IMAGES_PER_FRUIT = 500
DEFAULT_OUT_SIZE = (256, 256)
DEFAULT_OUTPUT_DIR = "dataset"

# Expanded search terms for each fruit category
FRUIT_SEARCH_TERMS = {
    "orange": [
        "fresh orange fruit", "orange citrus fruit", "ripe orange", "orange fruit white background",
        "orange slice", "valencia orange", "navel orange", "orange fruit close up",
        "organic orange fruit", "single orange fruit", "orange fruit photography", "orange isolated",
        "blood orange", "mandarin orange", "orange segments", "peeled orange",
        "orange juice orange", "whole orange", "orange peel", "orange tree fruit",
        "cara cara orange", "sweet orange", "tangerine orange", "citrus sinensis"
    ],
    "strawberry": [
        "fresh strawberry fruit", "ripe strawberry", "strawberry berry fruit", "strawberry white background",
        "single strawberry", "organic strawberry", "red strawberry fruit", "strawberry fruit close up",
        "fresh picked strawberry", "strawberry fruit photography", "strawberry isolated", "garden strawberry",
        "wild strawberry", "strawberry macro", "strawberry leaf", "alpine strawberry",
        "june bearing strawberry", "everbearing strawberry", "fragaria strawberry", "strawberry plant fruit",
        "day neutral strawberry", "woodland strawberry", "strawberry crown", "harvest strawberry"
    ],
    "banana": [
        "fresh banana fruit", "ripe banana", "yellow banana fruit", "banana white background",
        "single banana", "organic banana", "banana fruit close up", "peeled banana",
        "banana bunch", "cavendish banana", "banana fruit photography", "banana isolated",
        "green banana", "plantain banana", "lady finger banana", "red banana",
        "banana hand", "tropical banana", "musa banana", "banana tree fruit",
        "dwarf banana", "cooking banana", "dessert banana", "banana finger"
    ],
    "pineapple": [
        "fresh pineapple fruit", "ripe pineapple", "pineapple tropical fruit", "pineapple white background",
        "whole pineapple", "organic pineapple", "pineapple fruit close up", "golden pineapple",
        "pineapple crown", "sweet pineapple", "pineapple fruit photography", "pineapple isolated",
        "pineapple slice", "pineapple core", "ananas pineapple", "del monte pineapple",
        "hawaiian pineapple", "costa rican pineapple", "pineapple chunks", "tropical ananas",
        "smooth cayenne pineapple", "queen pineapple", "red spanish pineapple", "sugarloaf pineapple"
    ],
    "blackberry": [
        "fresh blackberry fruit", "ripe blackberry", "blackberry berry fruit", "blackberry white background",
        "organic blackberry", "dark blackberry", "blackberry fruit close up", "wild blackberry",
        "blackberry cluster", "blackberry fruit photography", "blackberry isolated", "bramble blackberry",
        "dewberry", "boysenberry", "marionberry", "tayberry",
        "rubus blackberry", "thornless blackberry", "blackberry bush fruit", "aggregate blackberry",
        "black raspberry", "caneberry", "blackberry drupelets", "summer blackberry"
    ]
}

# Search engines to use
SEARCH_ENGINES = {
    'google': {
        'base_url': 'https://www.google.com/search',
        'params': {
            'tbm': 'isch',
            'tbs': 'isz:m,itp:photo',  # Medium size, photos only
            'safe': 'active'
        }
    },
    'bing': {
        'base_url': 'https://www.bing.com/images/search',
        'params': {
            'qft': '+filterui:aspect-square+filterui:imagesize-medium',
            'FORM': 'IRFLTR'
        }
    },
    'duckduckgo': {
        'base_url': 'https://duckduckgo.com/',
        'params': {
            'q': '',
            'iax': 'images',
            'ia': 'images'
        }
    }
}

# Image quality filters - very lenient
MIN_AR = 0.2    # min aspect ratio (w/h)
MAX_AR = 5.0    # max aspect ratio (w/h)
MIN_SIZE = 64   # minimum image dimension
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Network settings
REQUEST_TIMEOUT = 8  # Reduced timeout
MAX_RETRIES = 2      # Fewer retries for faster testing
DELAY_BETWEEN_REQUESTS = 0.2  # Faster processing


class DatasetCollector:
    """Enhanced dataset collector using curated image sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = self._create_session()
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'total_attempted': 0,
            'total_downloaded': 0,
            'total_filtered': 0,
            'total_duplicates': 0,
            'total_errors': 0
        }
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('dataset_collector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def ensure_folder(self, path: Path):
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)
    
    def compute_image_hashes(self, image: Image.Image) -> Dict[str, str]:
        """Compute multiple hash types for better duplicate detection."""
        try:
            return {
                'phash': str(imagehash.phash(image)),
                'dhash': str(imagehash.dhash(image)),
                'whash': str(imagehash.whash(image))
            }
        except Exception:
            return {}
    
    def is_duplicate(self, image_hashes: Dict[str, str], seen_hashes: Set[str]) -> bool:
        """Check if image is duplicate using multiple hash methods."""
        for hash_type, hash_value in image_hashes.items():
            if hash_value in seen_hashes:
                return True
        return False
    
    def search_images(self, search_term: str, search_engine: str = 'google', max_results: int = 50) -> List[str]:
        """Search for images using the specified search engine."""
        try:
            engine_config = SEARCH_ENGINES.get(search_engine, SEARCH_ENGINES['google'])
            
            # Set up headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Referer': 'https://www.google.com/',
            }
            
            # Prepare search parameters
            params = engine_config['params'].copy()
            params['q'] = search_term
            
            self.logger.debug(f"Searching {search_engine} for: {search_term}")
            
            response = requests.get(
                engine_config['base_url'], 
                params=params, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Search failed with status {response.status_code}")
                return []
            
            # Parse image URLs from the response
            return self.extract_image_urls(response.text, search_engine, max_results)
            
        except Exception as e:
            self.logger.error(f"Search error for '{search_term}': {str(e)}")
            return []
    
    def extract_image_urls(self, html_content: str, search_engine: str, max_results: int) -> List[str]:
        """Extract image URLs from search engine HTML."""
        image_urls = []
        
        if not HAS_BS4:
            self.logger.error("BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")
            return []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            if search_engine == 'google':
                # Google Images structure
                img_tags = soup.find_all('img')
                for img in img_tags:
                    src = img.get('src') or img.get('data-src')
                    if src and self.is_valid_image_url(src):
                        image_urls.append(src)
                        if len(image_urls) >= max_results:
                            break
            
            elif search_engine == 'bing':
                # Bing Images structure  
                img_tags = soup.find_all('img', class_='mimg')
                for img in img_tags:
                    src = img.get('src') or img.get('data-src')
                    if src and self.is_valid_image_url(src):
                        image_urls.append(src)
                        if len(image_urls) >= max_results:
                            break
            
            self.logger.debug(f"Extracted {len(image_urls)} image URLs from {search_engine}")
            return image_urls[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error extracting URLs: {str(e)}")
            return []
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        if not url or url.startswith('data:'):
            return False
        
        # Convert relative URLs to absolute
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            return False  # Skip relative URLs that need base domain
        
        # Check for image extensions or image-related domains
        image_indicators = ['.jpg', '.jpeg', '.png', '.webp', '.gif', 'image', 'photo', 'pic']
        return any(indicator in url.lower() for indicator in image_indicators)
    
    def get_search_images(self, category: str, offset: int = 0, batch_size: int = 50) -> List[Dict]:
        """Get images for a category by searching multiple search engines."""
        search_terms = FRUIT_SEARCH_TERMS.get(category.lower(), [])
        
        if not search_terms:
            self.logger.warning(f"No search terms found for category: {category}")
            return []
        
        all_urls = []
        
        # Randomize search terms order for diversity
        shuffled_terms = search_terms.copy()
        random.shuffle(shuffled_terms)
        
        # Calculate which terms to use based on offset
        terms_per_batch = max(1, len(search_terms) // 3)  # Use subset per batch
        start_term = (offset // batch_size) % len(search_terms)
        selected_terms = shuffled_terms[start_term:start_term + terms_per_batch]
        if len(selected_terms) < terms_per_batch:
            # Wrap around if needed
            selected_terms.extend(shuffled_terms[:terms_per_batch - len(selected_terms)])
        
        # Try multiple search engines and terms
        for engine in ['google', 'bing', 'duckduckgo']:
            for term in selected_terms:
                if len(all_urls) >= batch_size * 4:  # Get more extras to account for failures
                    break
                    
                # Add variation to search terms for more diversity
                if offset > 0:
                    term_variations = [
                        term,
                        f"{term} high quality",
                        f"{term} macro photography",
                        f"{term} studio shot"
                    ]
                    term = random.choice(term_variations)
                
                urls = self.search_images(term, engine, max_results=25)
                all_urls.extend(urls)
                
                # Add delay between searches to be respectful
                time.sleep(1.5)  # Slightly longer delay
        
        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))
        
        # Add randomization for final selection
        if len(unique_urls) > batch_size:
            # Take some from beginning (most relevant) and some random
            selected_urls = unique_urls[:batch_size//2]  # First half from top results
            remaining = unique_urls[batch_size//2:]
            random.shuffle(remaining)
            selected_urls.extend(remaining[:batch_size//2])  # Second half random
            unique_urls = selected_urls
        
        # Format for the expected structure
        results = [{"image": url} for url in unique_urls[:batch_size]]
        
        self.logger.info(f"Found {len(unique_urls)} unique images for {category}, returning {len(results)}")
        return results
        end_idx = start_idx + batch_size
        
        if start_idx >= len(all_variations):
            # If we've exhausted variations, start over with new timestamps
            time.sleep(0.1)  # Small delay to get different timestamps
            all_variations = self.generate_url_variations(base_urls, category, 500)
            start_idx = 0
            end_idx = batch_size
        
        results = all_variations[start_idx:end_idx]
        self.logger.debug(f"Generated {len(results)} image URLs for {category} (offset: {offset})")
        return results
    
    def download_and_process_image(self, url: str) -> Optional[Image.Image]:
        """Download and validate a single image."""
        if not url or not url.startswith('http'):
            self.logger.debug(f"Invalid URL: {url}")
            return None
            
        try:
            self.logger.debug(f"Attempting to download: {url}")
            # Download with timeout
            response = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            self.logger.debug(f"Successfully downloaded from: {url} (status: {response.status_code})")
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                self.logger.debug(f"Image too large: {content_length} bytes")
                return None
            
            # Read content with size limit
            content = BytesIO()
            size = 0
            for chunk in response.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    self.logger.debug(f"Image too large during download: {size} bytes")
                    return None
                content.write(chunk)
            
            if size < 1000:  # Image too small (likely broken)
                self.logger.debug(f"Image too small: {size} bytes")
                return None
            
            content.seek(0)
            
            # Open and validate image
            try:
                image = Image.open(content)
                image.verify()  # Check if image is corrupted
            except Exception as e:
                self.logger.debug(f"Image verification failed: {e}")
                return None
            
            # Reopen for actual processing (verify() closes the file)
            content.seek(0)
            try:
                image = Image.open(content)
                if image.mode in ('RGBA', 'LA'):
                    # Convert RGBA to RGB with white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'RGBA':
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                    image = background
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                self.logger.debug(f"Image conversion failed: {e}")
                return None
            
            # Basic size validation
            w, h = image.size
            if min(w, h) < 50:
                self.logger.debug(f"Image too small: {w}x{h}")
                return None
                
            return image
            
        except requests.exceptions.Timeout:
            self.logger.debug(f"Timeout downloading: {url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Request failed: {e}")
            return None
        except Exception as e:
            self.logger.debug(f"Unexpected error: {e}")
            return None
    
    def download_images_for_category(self, category: str, target_count: int, 
                                   output_dir: Path) -> int:
        """Download images for a specific category."""
        self.logger.info(f"Downloading '{category}' images (target: {target_count})")
        
        category_dir = output_dir / category
        self.ensure_folder(category_dir)
        
        # Load existing images
        existing_files = list(category_dir.glob("*.png"))
        existing_count = len(existing_files)
        
        if existing_count >= target_count:
            self.logger.info(f"Already have {existing_count} images for {category}")
            return existing_count
        
        seen_hashes: Set[str] = set()
        
        # Hash existing images
        for img_file in existing_files:
            try:
                with Image.open(img_file) as img:
                    hashes = self.compute_image_hashes(img)
                    for hash_value in hashes.values():
                        if hash_value:
                            seen_hashes.add(hash_value)
            except Exception:
                continue
        
        downloaded = existing_count
        offset = 0
        consecutive_failures = 0
        
        with tqdm(
            total=target_count, 
            initial=existing_count,
            desc=f"({category})", 
            ncols=100,
            unit="img"
        ) as pbar:
            
            while downloaded < target_count and consecutive_failures < 10:
                # Get images through search engines
                self.logger.debug(f"Getting search images for {category}, offset: {offset}")
                results = self.get_search_images(category, offset, 50)
                
                if not results:
                    consecutive_failures += 1
                    self.logger.warning(f"No results for {category} at offset {offset}, failure #{consecutive_failures}")
                    offset += 50
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    continue
                    
                self.logger.info(f"Got {len(results)} URLs for {category}, processing batch...")
                
                batch_downloaded = 0
                batch_processed = 0
                
                # Process images
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_url = {
                        executor.submit(self.download_and_process_image, item["image"]): item["image"]
                        for item in results
                    }
                    
                    for future in as_completed(future_to_url):
                        if downloaded >= target_count:
                            break
                        
                        url = future_to_url[future]
                        batch_processed += 1
                        self.stats['total_attempted'] += 1
                        image = future.result()
                        
                        if image is None:
                            self.stats['total_errors'] += 1
                            continue
                        
                        # Size and aspect ratio checks
                        w, h = image.size
                        if min(w, h) < MIN_SIZE:
                            self.stats['total_filtered'] += 1
                            continue
                        
                        ratio = w / h
                        if ratio < MIN_AR or ratio > MAX_AR:
                            self.stats['total_filtered'] += 1
                            continue
                        
                        # Duplicate check
                        try:
                            image_hashes = self.compute_image_hashes(image)
                            if image_hashes and self.is_duplicate(image_hashes, seen_hashes):
                                self.stats['total_duplicates'] += 1
                                continue
                            
                            # Add new hashes
                            for hash_value in image_hashes.values():
                                if hash_value:
                                    seen_hashes.add(hash_value)
                                    
                        except Exception:
                            pass  # Continue even if hashing fails
                        
                        # Resize and save
                        try:
                            image = image.resize(
                                self.config['output_size'], 
                                Image.Resampling.LANCZOS
                            )
                            
                            # Optional enhancement
                            if self.config.get('enhance_images', False):
                                image = image.filter(ImageFilter.SHARPEN)
                            
                            # Save image
                            filename = category_dir / f"{category}_{downloaded:05d}.png"
                            image.save(filename, "PNG", optimize=True)
                            
                            downloaded += 1
                            batch_downloaded += 1
                            self.stats['total_downloaded'] += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            self.logger.debug(f"Save failed: {e}")
                            continue
                
                if batch_downloaded == 0:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0  # Reset on success
                
                offset += 50
                time.sleep(DELAY_BETWEEN_REQUESTS)
        
        self.logger.info(f"Completed {category}: {downloaded} images downloaded")
        return downloaded
    
    def save_metadata(self, output_dir: Path, categories: List[str], results: Dict[str, int]):
        """Save collection metadata."""
        metadata = {
            'config': self.config,
            'categories': categories,
            'results': results,
            'statistics': self.stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_file}")
    
    def collect_dataset(self, categories: List[str]) -> Dict[str, int]:
        """Main collection process."""
        output_dir = Path(self.config['output_dir'])
        self.ensure_folder(output_dir)
        
        results = {}
        total_images = 0
        
        self.logger.info(f"Starting dataset collection for {len(categories)} categories")
        self.logger.info(f"Output directory: {output_dir}")
        
        for category in categories:
            try:
                count = self.download_images_for_category(
                    category, 
                    self.config['images_per_category'],
                    output_dir
                )
                results[category] = count
                total_images += count
                
            except KeyboardInterrupt:
                self.logger.info("Collection interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error collecting {category}: {e}")
                results[category] = 0
        
        # Save metadata
        self.save_metadata(output_dir, categories, results)
        
        self.logger.info(f"Collection complete! Total images: {total_images}")
        self.logger.info(f"Statistics: {self.stats}")
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Image Dataset Collector with Curated Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 data_collection_fixed.py
  python3 data_collection_fixed.py --categories apple orange banana --count 500
  python3 data_collection_fixed.py --output my_dataset --size 512 512
  python3 data_collection_fixed.py --enhance --verbose
        """
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        default=DEFAULT_FRUITS,
        help=f'Categories to collect (default: {DEFAULT_FRUITS})'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=DEFAULT_IMAGES_PER_FRUIT,
        help=f'Images per category (default: {DEFAULT_IMAGES_PER_FRUIT})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--size',
        nargs=2,
        type=int,
        default=DEFAULT_OUT_SIZE,
        metavar=('WIDTH', 'HEIGHT'),
        help=f'Output image size (default: {DEFAULT_OUT_SIZE[0]} {DEFAULT_OUT_SIZE[1]})'
    )
    
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Apply image enhancement (sharpening)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging (very detailed)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging level
    if args.debug:
        logging.getLogger('dataset_collector').setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger('dataset_collector').setLevel(logging.INFO)
    
    # Configuration
    config = {
        'output_dir': args.output,
        'images_per_category': args.count,
        'output_size': tuple(args.size),
        'enhance_images': args.enhance
    }
    
    print("=" * 60)
    print("Enhanced Image Dataset Collector (Curated Sources)")
    print("=" * 60)
    print(f"Categories: {args.categories}")
    print(f"Images per category: {args.count}")
    print(f"Output size: {args.size[0]}x{args.size[1]}")
    print(f"Output directory: {args.output}")
    print(f"Image enhancement: {'ON' if args.enhance else 'OFF'}")
    print("=" * 60)
    
    # Create collector and run
    collector = DatasetCollector(config)
    
    try:
        results = collector.collect_dataset(args.categories)
        
        print("\n" + "=" * 60)
        print("Collection Results:")
        for category, count in results.items():
            print(f"  {category}: {count} images")
        print(f"Total: {sum(results.values())} images")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nCollection cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logging.getLogger('dataset_collector').exception("Collection failed")


if __name__ == "__main__":
    main()