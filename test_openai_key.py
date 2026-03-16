#!/usr/bin/env python3
"""
Test multiple OpenAI API keys to find working ones.

Usage:
    python test_openai_key.py
"""

from openai import OpenAI
import time

# ============================================================
# ADD YOUR API KEYS HERE
# ============================================================
API_KEYS = [
    "sk-abcdef1234567890abcdef1234567890abcdef12",
"sk-1234567890abcdef1234567890abcdef12345678",
"sk-abcdefabcdefabcdefabcdefabcdefabcdef12",
"sk-7890abcdef7890abcdef7890abcdef7890abcd",
"sk-1234abcd1234abcd1234abcd1234abcd1234abcd",
"sk-abcd1234abcd1234abcd1234abcd1234abcd1234",
"sk-5678efgh5678efgh5678efgh5678efgh5678efgh",
"sk-efgh5678efgh5678efgh5678efgh5678efgh5678",
"sk-ijkl1234ijkl1234ijkl1234ijkl1234ijkl1234",
"sk-mnop5678mnop5678mnop5678mnop5678mnop5678",
"sk-qrst1234qrst1234qrst1234qrst1234qrst1234",
"sk-uvwx5678uvwx5678uvwx5678uvwx5678uvwx5678",
"sk-1234ijkl1234ijkl1234ijkl1234ijkl1234ijkl",
"sk-5678mnop5678mnop5678mnop5678mnop5678mnop",
"sk-qrst5678qrst5678qrst5678qrst5678qrst5678",
"sk-uvwx1234uvwx1234uvwx1234uvwx1234uvwx1234",
"sk-1234abcd5678efgh1234abcd5678efgh1234abcd",
"sk-5678ijkl1234mnop5678ijkl1234mnop5678ijkl",
"sk-abcdqrstefghuvwxabcdqrstefghuvwxabcdqrst",
"sk-ijklmnop1234qrstijklmnop1234qrstijklmnop",
"sk-1234uvwx5678abcd1234uvwx5678abcd1234uvwx",
"sk-efghijkl5678mnopabcd1234efghijkl5678mnop",
"sk-mnopqrstuvwxabcdmnopqrstuvwxabcdmnopqrst",
"sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop",
"sk-abcd1234efgh5678abcd1234efgh5678abcd1234",
"sk-1234ijklmnop5678ijklmnop1234ijklmnop5678",
"sk-qrstefghuvwxabcdqrstefghuvwxabcdqrstefgh",
"sk-uvwxijklmnop1234uvwxijklmnop1234uvwxijkl",
"sk-abcd5678efgh1234abcd5678efgh1234abcd5678",
"sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop",
"sk-1234qrstuvwxabcd1234qrstuvwxabcd1234qrst",
"sk-efghijklmnop5678efghijklmnop5678efghijkl",
"sk-mnopabcd1234efghmnopabcd1234efghmnopabcd",
"sk-ijklqrst5678uvwxijklqrst5678uvwxijklqrst",
"sk-1234ijkl5678mnop1234ijkl5678mnop1234ijkl",
"sk-abcdqrstefgh5678abcdqrstefgh5678abcdqrst",
"sk-ijklmnopuvwx1234ijklmnopuvwx1234ijklmnop",
"sk-efgh5678abcd1234efgh5678abcd1234efgh5678",
"sk-mnopqrstijkl5678mnopqrstijkl5678mnopqrst",
"sk-1234uvwxabcd5678uvwxabcd1234uvwxabcd5678",
"sk-ijklmnop5678efghijklmnop5678efghijklmnop",
"sk-abcd1234qrstuvwxabcd1234qrstuvwxabcd1234",
"sk-1234efgh5678ijkl1234efgh5678ijkl1234efgh",
"sk-5678mnopqrstuvwx5678mnopqrstuvwx5678mnop",
"sk-abcdijkl1234uvwxabcdijkl1234uvwxabcdijkl",
"sk-ijklmnopabcd5678ijklmnopabcd5678ijklmnop",
"sk-1234efghqrstuvwx1234efghqrstuvwx1234efgh",
"sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl",
"sk-abcd1234efgh5678abcd1234efgh5678abcd1234",
"sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop"
]


def test_single_key(api_key: str, index: int):
    """Test a single OpenAI API key."""
    
    key_preview = f"{api_key[:15]}...{api_key[-4:]}" if len(api_key) > 20 else api_key
    
    print(f"\n{'='*70}")
    print(f"� Testing Key #{index + 1}: {key_preview}")
    print(f"{'='*70}")
    
    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Make a simple test request
        print("   ⏳ Sending test request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with exactly: 'Working!'"}
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        # Extract response
        message = response.choices[0].message.content
        
        print(f"   ✅ SUCCESS! API key is VALID and working!")
        print(f"   📝 Response: {message}")
        print(f"   💰 Tokens used: {response.usage.total_tokens}")
        
        return {
            'key': api_key,
            'key_preview': key_preview,
            'status': 'VALID',
            'response': message,
            'tokens': response.usage.total_tokens
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ FAILED: {error_msg[:100]}")
        
        # Categorize error
        if "incorrect_api_key" in error_msg.lower() or "invalid" in error_msg.lower():
            error_type = "INVALID_KEY"
        elif "rate_limit" in error_msg.lower():
            error_type = "RATE_LIMITED"
        elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            error_type = "NO_QUOTA"
        elif "permission" in error_msg.lower():
            error_type = "NO_PERMISSION"
        else:
            error_type = "UNKNOWN_ERROR"
        
        return {
            'key': api_key,
            'key_preview': key_preview,
            'status': 'INVALID',
            'error_type': error_type,
            'error': error_msg[:200]
        }


def test_all_keys():
    """Test all API keys and return results."""
    
    print("\n" + "="*70)
    print("🚀 TESTING MULTIPLE OPENAI API KEYS")
    print("="*70)
    print(f"\n📊 Total keys to test: {len(API_KEYS)}")
    
    results = []
    
    for i, key in enumerate(API_KEYS):
        if not key or key.startswith("sk-key-") or len(key) < 20:
            print(f"\n⚠️  Skipping Key #{i + 1}: Placeholder detected")
            continue
        
        result = test_single_key(key, i)
        results.append(result)
        
        # Small delay to avoid rate limits
        if i < len(API_KEYS) - 1:
            time.sleep(0.5)
    
    return results


def print_summary(results):
    """Print summary of all test results."""
    
    print("\n\n" + "="*70)
    print("📋 SUMMARY REPORT")
    print("="*70)
    
    valid_keys = [r for r in results if r['status'] == 'VALID']
    invalid_keys = [r for r in results if r['status'] == 'INVALID']
    
    print(f"\n✅ Valid Keys: {len(valid_keys)}/{len(results)}")
    print(f"❌ Invalid Keys: {len(invalid_keys)}/{len(results)}")
    
    if valid_keys:
        print("\n" + "="*70)
        print("✅ WORKING KEYS:")
        print("="*70)
        for i, result in enumerate(valid_keys, 1):
            print(f"\n{i}. {result['key_preview']}")
            print(f"   Status: ✅ VALID")
            print(f"   Response: {result['response']}")
            print(f"   Tokens: {result['tokens']}")
            print(f"   Full Key: {result['key']}")
    
    if invalid_keys:
        print("\n" + "="*70)
        print("❌ INVALID KEYS:")
        print("="*70)
        for i, result in enumerate(invalid_keys, 1):
            print(f"\n{i}. {result['key_preview']}")
            print(f"   Status: ❌ {result['error_type']}")
            print(f"   Error: {result['error'][:150]}...")
    
    if valid_keys:
        print("\n" + "="*70)
        print("🎯 RECOMMENDED ACTION:")
        print("="*70)
        print(f"\nUse this command to set your working API key:")
        print(f"\nexport OPENAI_API_KEY=\"{valid_keys[0]['key']}\"")
        print(f"\nOr add to ~/.bash_profile:")
        print(f"echo 'export OPENAI_API_KEY=\"{valid_keys[0]['key']}\"' >> ~/.bash_profile")
        print(f"source ~/.bash_profile")


if __name__ == "__main__":
    results = test_all_keys()
    
    if results:
        print_summary(results)
    else:
        print("\n❌ No valid API keys found in the array!")
        print("\n📝 Instructions:")
        print("1. Open test_openai_key.py")
        print("2. Replace the placeholder keys in API_KEYS array with your actual keys")
        print("3. Run the script again")
    
    print("\n" + "="*70)
