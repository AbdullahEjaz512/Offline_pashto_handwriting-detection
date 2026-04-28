import requests
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.pipeline import FullPagePashtoRecognition

def download_image(url, filename):
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def test_pipeline():
    # 1. Setup Test Images
    test_data = [
        ("https://dfzljdn9uc3pi.cloudfront.net/2024/cs-1925/1/fig-2-2x.jpg", "test_neat.jpg"),
        ("https://i.redd.it/pashto-english-request-v0-dc2qjnfi886g1.jpg", "test_messy.jpg"),
        ("https://www.researchgate.net/profile/Sikandar-Khan-12/publication/328754005/figure/fig1/AS:689626508218370@1541431057434/Pashto-Handwritten-transformed-invariant-words.png", "test_words.png"),
        ("https://upload.wikimedia.org/wikipedia/commons/9/9e/Pashto_Qamus1.jpg", "test_calligraphy.jpg")
    ]
    
    os.makedirs("test_wild", exist_ok=True)
    
    # 2. Initialize Pipeline
    print("\n--- Initializing Pipeline ---")
    pipeline = FullPagePashtoRecognition()
    
    # 3. Run Tests
    for url, filename in test_data:
        path = os.path.join("test_wild", filename)
        if download_image(url, path):
            print(f"\n--- Testing: {filename} ---")
            results = pipeline.process_page(path)
            
            print(f"Extracted {len(results)} lines:")
            for i, line in enumerate(results[:10]): # Show first 10 lines
                print(f"  Line {i+1}: {line}")
            if len(results) > 10:
                print(f"  ... and {len(results)-10} more lines.")
        print("-" * 30)

if __name__ == "__main__":
    test_pipeline()
