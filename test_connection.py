"""
Test script to verify all components are working correctly
"""
import sys
import config

def test_qdrant_connection():
    """Test Qdrant connection"""
    print("Testing Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        collections = client.get_collections()
        print(f"[OK] Qdrant is connected! Collections: {collections}")
        return True
    except Exception as e:
        print(f"[FAIL] Qdrant connection failed: {e}")
        return False

def test_gemini_api():
    """Test Gemini API key"""
    print("\nTesting Gemini API...")
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_api_key_here":
        print("[FAIL] GEMINI_API_KEY is not set or still has placeholder value")
        print("   Please update the .env file with your actual API key")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        response = model.generate_content("Say 'Hello, I am working!'")
        print(f"[OK] Gemini API is working! Response: {response.text}")
        return True
    except Exception as e:
        print(f"[FAIL] Gemini API test failed: {e}")
        return False

def test_embedding_model():
    """Test Sentence Transformer model"""
    print("\nTesting Embedding Model...")
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        test_text = "This is a test sentence."
        embedding = encoder.encode(test_text)
        print(f"[OK] Embedding model is working! Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding model test failed: {e}")
        return False

def test_vector_db():
    """Test VectorDB class"""
    print("\nTesting VectorDB class...")
    try:
        from vector_store import VectorDB
        db = VectorDB()
        print(f"[OK] VectorDB initialized successfully!")
        
        # Test adding a document
        test_text = "This is a test document.\n\nIt has multiple paragraphs.\n\nThis is the third paragraph."
        num_chunks = db.add_document(test_text, metadata={'source': 'test'})
        print(f"[OK] Added test document with {num_chunks} chunks")
        
        # Test search
        results = db.search("test document", k=2)
        print(f"[OK] Search returned {len(results)} results")
        if results:
            print(f"   Top result score: {results[0]['score']:.4f}")
        
        return True
    except Exception as e:
        print(f"[FAIL] VectorDB test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Chatbot System Test")
    print("=" * 60)
    
    results = []
    results.append(("Qdrant Connection", test_qdrant_connection()))
    results.append(("Gemini API", test_gemini_api()))
    results.append(("Embedding Model", test_embedding_model()))
    results.append(("VectorDB", test_vector_db()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n[SUCCESS] All tests passed! Your RAG chatbot is ready to use!")
        print("\nNext steps:")
        print("1. Ingest a document: python main.py ingest <path_to_pdf_or_image>")
        print("2. Start chatting: python main.py chat")
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)
