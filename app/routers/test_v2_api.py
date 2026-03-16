"""
Quick Test Script for AI Mashup V2 API

This script helps verify that the new V2 API is working correctly.
Run this after starting the FastAPI server.

Usage:
    python -m app.routers.test_v2_api
"""

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v2/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ LLM Providers: {', '.join(data['llm_providers_available'])}")
        print(f"✅ Operations Count: {data['operations_count']}")
        print(f"✅ Utils Modules: {len(data['utils_modules'])}")
        
        if data.get('errors'):
            print(f"⚠️  Warnings: {data['errors']}")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_operations_list():
    """Test operations listing."""
    print("\n" + "="*60)
    print("TEST 2: List Operations")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v2/operations")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Total operations available: {data['count']}")
        print(f"✅ Sample operations: {', '.join(data['operations'][:5])}...")
        
        return True
    except Exception as e:
        print(f"❌ Operations list failed: {e}")
        return False


def test_old_endpoint_still_works():
    """Verify old endpoint is still accessible."""
    print("\n" + "="*60)
    print("TEST 3: Old Endpoint (Backward Compatibility)")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Old /health endpoint still works")
        print(f"✅ Status: {data['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Old endpoint check failed: {e}")
        return False


def test_create_mashup_validation():
    """Test mashup creation endpoint validation (without actual files)."""
    print("\n" + "="*60)
    print("TEST 4: Mashup Creation Endpoint Validation")
    print("="*60)
    
    try:
        # This should fail without files, but endpoint should be reachable
        response = requests.post(f"{BASE_URL}/api/v2/ai-mashup")
        
        if response.status_code == 422:
            print("✅ Endpoint is accessible (validation working as expected)")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Endpoint validation failed: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    deps = {
        'fastapi': 'FastAPI',
        'pydantic': 'Pydantic',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'numpy': 'NumPy',
        'openai': 'OpenAI (optional)',
        'anthropic': 'Anthropic (optional)',
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            if 'optional' in name.lower():
                print(f"⚠️  {name} (optional - install if needed)")
            else:
                print(f"❌ {name} - MISSING!")
                all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AI MASHUP V2 API - INTEGRATION TEST")
    print("="*60)
    print("\nTesting new V2 endpoints...")
    print("Make sure the FastAPI server is running: uvicorn app.main:app --reload")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Some required dependencies are missing!")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Wait for user
    input("\nPress Enter when server is running...")
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Operations List", test_operations_list()))
    results.append(("Old Endpoint", test_old_endpoint_still_works()))
    results.append(("Validation", test_create_mashup_validation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! V2 API is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='sk-...'")
        print("2. Try creating a mashup with two audio files")
        print("3. See API_V2_DOCS.md for complete documentation")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()
