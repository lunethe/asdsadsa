"""
Test script - run against the API server or locally.
Usage:
  python test.py                    # Test against running server
  python test.py --local            # Test locally (loads model)
"""

import argparse
import json
import requests
import sys


SAMPLE_TEXT = """Apples are one of the most widely consumed fruits in the world, 
and their popularity is well-deserved. With their satisfying crunch and diverse 
flavor profiles, apples can be enjoyed fresh as a snack, baked into pies and 
desserts, or pressed into juice and cider. There are thousands of varieties, 
ranging from intensely sweet to sharply tart. Beyond their taste, apples offer 
significant health benefits. They are rich in dietary fiber, vitamins, and 
antioxidants that help protect cells from oxidative damage. Research suggests 
that regular apple consumption may support digestive health and cardiovascular 
function. Additionally, apples have deep cultural significance, appearing 
throughout history in mythology and folklore as symbols of knowledge, temptation, 
and discovery. From Newton's legendary falling apple to the Garden of Eden, 
this humble fruit has played a pivotal role in shaping human understanding 
of the world."""


def test_api(url: str = "http://localhost:8000"):
    """Test against running server."""
    print(f"Testing against {url}...")
    
    # Health check
    r = requests.get(f"{url}/health")
    print(f"Health: {r.json()}")
    
    # Humanize
    payload = {
        "text": SAMPLE_TEXT,
        "lex_diversity": 60,
        "order_diversity": 60,
        "sent_interval": 3,
        "post_process_enabled": True,
    }
    
    print("\nSending humanize request...")
    r = requests.post(f"{url}/humanize", json=payload)
    
    if r.status_code == 200:
        data = r.json()
        print(f"\n{'='*60}")
        print(f"Model: {data['model']}")
        print(f"Time: {data['processing_time']}s")
        print(f"Words: {data['original_words']} → {data['humanized_words']}")
        print(f"{'='*60}")
        print(f"\nORIGINAL:\n{data['original'][:200]}...")
        print(f"\nHUMANIZED:\n{data['humanized']}")
        print(f"\n{'='*60}")
        print("Copy the HUMANIZED text above and test it on:")
        print("  - https://copyleaks.com/ai-content-detector")
        print("  - https://gptzero.me")
        print("  - https://originality.ai")
    else:
        print(f"Error: {r.status_code} - {r.text}")


def test_local():
    """Test locally without server."""
    print("Loading model locally...")
    from app.main import paraphraser, post_process
    
    print(f"Paraphrasing with DIPPER (lex=60, order=60)...")
    result = paraphraser.paraphrase(SAMPLE_TEXT, lex_diversity=60, order_diversity=60)
    
    print(f"\nDIPPER output:\n{result}\n")
    
    processed = post_process(result)
    print(f"Post-processed:\n{processed}\n")
    
    print("Copy the text above and test it on detector sites.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Test locally")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    
    if args.local:
        test_local()
    else:
        test_api(args.url)
