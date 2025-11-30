"""Quick test of the RAG Chatbot components"""
import os
import sys

print("="*60)
print("RAG Chatbot Setup Test")
print("="*60)

# Test 1: Check .env
print("\n[1/4] Checking .env file...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and api_key != 'your_api_key_here':
        print("PASS - API key configured")
    else:
        print("FAIL - API key not set")
except Exception as e:
    print(f"FAIL - {e}")

# Test 2: Check Qdrant
print("\n[2/4] Checking Qdrant...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333, timeout=5)
    client.get_collections()
    print("PASS - Qdrant is running")
except Exception as e:
    print(f"FAIL - Qdrant not accessible: {e}")
    print("       Please start Docker Desktop and run: docker-compose up -d")

# Test 3: Check Gemini API
print("\n[3/4] Checking Gemini API...")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello")
    print(f"PASS - Gemini API working: {response.text[:30]}...")
except Exception as e:
    print(f"FAIL - {e}")

# Test 4: Check dependencies
print("\n[4/4] Checking key dependencies...")
try:
    import paddleocr
    import sentence_transformers
    print("PASS - All dependencies installed")
except Exception as e:
    print(f"FAIL - Missing dependencies: {e}")

print("\n" + "="*60)
print("Setup test complete!")
print("="*60)
