"""
Test RAG core functionality without OCR
"""
import sys

def test_qdrant():
    """Test Qdrant connection"""
    print("1. Testing Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        import config
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        collections = client.get_collections()
        print(f"   ✓ Qdrant connected! Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

def test_gemini():
    """Test Gemini API"""
    print("\n2. Testing Gemini API...")
    try:
        import google.generativeai as genai
        import config
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.LLM_MODEL_NAME)
        response = model.generate_content("Say 'OK'")
        print(f"   ✓ Gemini working! Response: {response.text[:50]}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

def test_embeddings():
    """Test embeddings"""
    print("\n3. Testing Sentence Transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        import config
        encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        embedding = encoder.encode("test")
        print(f"   ✓ Embeddings working! Dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

def test_vector_store():
    """Test vector store"""
    print("\n4. Testing VectorDB...")
    try:
        from vector_store import VectorDB
        db = VectorDB()
        
        # Add test document
        test_text = """
        This is a test document about machine learning.
        Machine learning is a subset of artificial intelligence.
        It involves training models on data to make predictions.
        """
        num_chunks = db.add_document(test_text, metadata={'source': 'test'})
        print(f"   ✓ Added {num_chunks} chunks to VectorDB")
        
        # Test search
        results = db.search("machine learning", k=2)
        print(f"   ✓ Search returned {len(results)} results")
        if results:
            print(f"   ✓ Top result score: {results[0]['score']:.4f}")
        
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chatbot():
    """Test chatbot"""
    print("\n5. Testing RAG Chatbot...")
    try:
        from chatbot import RAGChatbot
        bot = RAGChatbot()
        
        # Ask a question
        response = bot.query("What is machine learning?")
        print(f"   ✓ Chatbot response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("RAG CORE FUNCTIONALITY TEST (No OCR)")
    print("=" * 60)
    
    tests = [
        test_qdrant,
        test_gemini,
        test_embeddings,
        test_vector_store,
        test_chatbot
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"   ✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED! RAG system is working!")
        print("\nYou can now:")
        print("  - Ingest documents: python main.py ingest <file>")
        print("  - Chat: python main.py chat")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        sys.exit(1)
