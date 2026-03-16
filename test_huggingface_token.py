#!/usr/bin/env python3
"""
Test Hugging Face API token - completely FREE!

Get your FREE token at: https://huggingface.co/settings/tokens

Usage:
    python test_huggingface_token.py
"""

import os
import requests
import time
import json

# Get token from environment
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')

def test_huggingface_token():
    """Test if Hugging Face token is valid and working."""
    
    print("=" * 70)
    print("🤗 HUGGING FACE TOKEN TEST (FREE!)")
    print("=" * 70)
    
    if not HUGGINGFACE_TOKEN:
        print("\n❌ ERROR: HUGGINGFACE_TOKEN environment variable not set!")
        print("\n📝 To get a FREE token:")
        print("1. Go to https://huggingface.co/join (sign up free)")
        print("2. Go to https://huggingface.co/settings/tokens")
        print("3. Click 'New token' → Select 'Read' permission")
        print("4. Copy your token (starts with 'hf_...')")
        print("\n💻 Then set it:")
        print('export HUGGINGFACE_TOKEN="hf_your_token_here"')
        return False
    
    print(f"\n✓ Found token: {HUGGINGFACE_TOKEN[:15]}...{HUGGINGFACE_TOKEN[-4:]}")
    
    # Test 1: Validate token with whoami endpoint
    print("\n🔍 Step 1: Validating token...")
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"   ✅ Token is valid!")
            print(f"   👤 Username: {user_info.get('name', 'Unknown')}")
            print(f"   📧 Email: {user_info.get('email', 'N/A')}")
        elif response.status_code == 401:
            print("   ❌ Token is INVALID or expired!")
            print("\n🔑 Please generate a new token:")
            print("   1. Go to https://huggingface.co/settings/tokens")
            print("   2. Delete the old token")
            print("   3. Create a new one with 'Read' permission")
            return False
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error validating token: {e}")
        return False
    
    # Test 2: Test with a simple inference API call
    print("\n🔍 Step 2: Testing Inference API...")
    print("   Using model: distilbert-base-uncased-finetuned-sst-2-english")
    print("   (Sentiment analysis model - small and fast)")
    
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    payload = {
        "inputs": "I love using Hugging Face!"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n   ⏳ Sending test request (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 503:
                # Model is loading
                result = response.json()
                wait_time = result.get('estimated_time', 20)
                print(f"   ⏳ Model is loading... waiting {wait_time} seconds")
                time.sleep(wait_time)
                continue
            
            if response.status_code == 401:
                print("   ❌ Authentication failed! Token may be invalid.")
                return False
            
            if response.status_code == 403:
                print("   ❌ Access denied! Token may not have correct permissions.")
                print("   Make sure your token has 'Read' or 'Inference' permission.")
                return False
            
            response.raise_for_status()
            result = response.json()
            
            print("\n   ✅ SUCCESS! Inference API is working!")
            print(f"   📝 Response: {json.dumps(result, indent=2)[:200]}")
            print("\n🎉 Your FREE Hugging Face token is valid and ready to use!")
            print("\n💡 You can now use Hugging Face in MAI Mashup Creator:")
            print("   1. Make sure server is running with mai-env:")
            print("      /opt/anaconda3/envs/mai-env/bin/uvicorn app.main:app --reload")
            print("   2. Open http://localhost:8000")
            print("   3. Select '🆓 Hugging Face (FREE!)' as AI Provider")
            print("   4. Create unlimited FREE mashups!")
            
            return True
            
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print("   Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("\n❌ Model took too long to respond")
                print("   This can happen with free tier. Try again in a few minutes.")
                return False
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)[:200]}")
            if attempt < max_retries - 1:
                print(f"   Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print("\n🔧 Troubleshooting:")
                print("1. Check your internet connection")
                print("2. Verify token at https://huggingface.co/settings/tokens")
                print("3. Make sure token has 'Read' or 'Inference' permission")
                print("4. Try generating a new token")
                return False
    
    return False


if __name__ == "__main__":
    success = test_huggingface_token()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ READY TO GO!")
        print("\nYour FREE Hugging Face setup is complete.")
        print("Start creating mashups with zero cost!")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nFollow the instructions above to fix the issue.")
        print("Get help at: https://huggingface.co/docs/api-inference")
    print("=" * 70)ython3
"""
Test Hugging Face API token - completely FREE!

Get your FREE token at: https://huggingface.co/settings/tokens

Usage:
    python test_huggingface_token.py
"""

import os
import requests
import time

# Get token from environment
HUGGINGFACE_TOKEN = ""

def test_huggingface_token():
    """Test if Hugging Face token is valid and working."""
    
    print("=" * 70)
    print("🤗 HUGGING FACE TOKEN TEST (FREE!)")
    print("=" * 70)
    
    if not HUGGINGFACE_TOKEN:
        print("\n❌ ERROR: HUGGINGFACE_TOKEN environment variable not set!")
        print("\n📝 To get a FREE token:")
        print("1. Go to https://huggingface.co/join (sign up free)")
        print("2. Go to https://huggingface.co/settings/tokens")
        print("3. Click 'New token' → Select 'Read' permission")
        print("4. Copy your token (starts with 'hf_...')")
        print("\n💻 Then set it:")
        print('export HUGGINGFACE_TOKEN="hf_your_token_here"')
        return False
    
    print(f"\n✓ Found token: {HUGGINGFACE_TOKEN[:15]}...{HUGGINGFACE_TOKEN[-4:]}")
    
    # Test with a simple, widely available model
    model = "google/flan-t5-base"
    print(f"\n🔍 Testing with model: {model}")
    print("   This is a FREE model, completely free to use!")
    
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    
    # Simple test prompt (format for T5 model)
    prompt = "Translate to English: 'Bonjour'"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50
        }
    }
    
    print("\n⏳ Sending test request...")
    print("   (If model is loading, this may take 10-30 seconds)")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 503:
                print(f"\n⏳ Model is loading... waiting 10 seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
                continue
            
            if response.status_code == 401:
                print("\n❌ ERROR: Invalid token!")
                print("\n🔑 Your token is not valid or doesn't have the right permissions.")
                print("   Go to https://huggingface.co/settings/tokens and:")
                print("   1. Generate a new token")
                print("   2. Make sure it has 'Read' permission")
                print("   3. Copy and use the new token")
                return False
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response
            if isinstance(result, list) and len(result) > 0:
                message = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                message = result.get('generated_text', '')
            else:
                message = str(result)
            
            print("\n✅ SUCCESS! Hugging Face API is working!")
            print(f"\n📝 Response from {model}:")
            print(f"   {message}")
            print("\n🎉 Your FREE Hugging Face token is valid and ready to use!")
            print("\n💡 You can now use Hugging Face in MAI Mashup Creator:")
            print("   1. Start server: uvicorn app.main:app --reload")
            print("   2. Open http://localhost:8000")
            print("   3. Select '🆓 Hugging Face (FREE!)' as AI Provider")
            print("   4. Create unlimited FREE mashups!")
            
            return True
            
        except requests.exceptions.Timeout:
            print(f"\n⏰ Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print("   Retrying...")
                time.sleep(5)
            else:
                print("\n❌ Model took too long to respond")
                print("   This can happen with free tier. Try again in a few minutes.")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                print("\n🔧 Troubleshooting:")
                print("1. Check your internet connection")
                print("2. Verify token at https://huggingface.co/settings/tokens")
                print("3. Try generating a new token")
                print("4. Make sure token has 'Read' permission")
                return False
    
    return False


if __name__ == "__main__":
    success = test_huggingface_token()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ READY TO GO!")
        print("\nYour FREE Hugging Face setup is complete.")
        print("Start creating mashups with zero cost!")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nFollow the instructions above to get your FREE token.")
        print("It only takes 2 minutes and requires no credit card!")
    print("=" * 70)
