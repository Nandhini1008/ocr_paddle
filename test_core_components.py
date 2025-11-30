"""
Test core RAG components (Qdrant + Embeddings + Vector Store)
"""
import sys

print("=" * 60)
print("TESTING RAG CORE COMPONENTS")
print("=" * 60)

# Test 1: Qdrant
print("\n1. Testing Qdrant...")
try:
    from qdrant_client import QdrantClient
    import config
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    collections = client.get_collections()
    print(f"   ✓ Qdrant is running! ({len(collections.collections)} collections)")
    qdrant_ok = True
except Exception as e:
    print(f"   ✗ Qdrant failed: {e}")
    qdrant_ok = False

# Test 2: Embeddings
print("\n2. Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    import config
    print(f"   Loading model: {config.EMBEDDING_MODEL_NAME}")
    encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    test_embedding = encoder.encode("This is a test sentence")
    print(f"   ✓ Embeddings working! Dimension: {len(test_embedding)}")
    embeddings_ok = True
except Exception as e:
    print(f"   ✗ Embeddings failed: {e}")
    embeddings_ok = False

# Test 3: Vector Store
print("\n3. Testing VectorDB...")
try:
    from vector_store import VectorDB
    db = VectorDB()
    print("   ✓ VectorDB initialized")
    
    # Add test document
    test_doc = """
    Python is a high-level programming language.
    It is widely used for web development, data science, and machine learning.
    Python has a simple and readable syntax.
    Many developers love Python for its versatility.
    """
    
    num_chunks = db.add_document(test_doc, metadata={'source': 'test_doc', 'type': 'test'})
    print(f"   ✓ Added test document ({num_chunks} chunks)")
    
    # Test search
    results = db.search("programming language", k=3)
    print(f"   ✓ Search returned {len(results)} results")
    
    if results:
        print(f"   ✓ Top result score: {results[0]['score']:.4f}")
        print(f"   ✓ Top result text: {results[0]['text'][:60]}...")
    
    vectordb_ok = True
except Exception as e:
    print(f"   ✗ VectorDB failed: {e}")
    import traceback
    traceback.print_exc()
    vectordb_ok = False

# Test 4: Gemini API (optional)
print("\n4. Testing Gemini API...")
try:
    import google.generativeai as genai
    import config
    
    if not config.GEMINI_API_KEY:
        print("   ⚠ No API key found")
        gemini_ok = False
    else:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content('Say "Hello"')
        print(f"   ✓ Gemini working! Response: {response.text[:50]}")
        gemini_ok = True
except Exception as e:
    print(f"   ⚠ Gemini API issue (may need valid key): {str(e)[:100]}")
    gemini_ok = False

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"Qdrant:      {'✓ PASS' if qdrant_ok else '✗ FAIL'}")
print(f"Embeddings:  {'✓ PASS' if embeddings_ok else '✗ FAIL'}")
print(f"VectorDB:    {'✓ PASS' if vectordb_ok else '✗ FAIL'}")
print(f"Gemini API:  {'✓ PASS' if gemini_ok else '⚠ SKIP (API key issue)'}")

core_ok = qdrant_ok and embeddings_ok and vectordb_ok

print("\n" + "=" * 60)
if core_ok:
    print("✓ CORE RAG SYSTEM IS WORKING!")
    print("\nThe vector database and retrieval system are functional.")
    if not gemini_ok:
        print("\n⚠ Note: Gemini API needs a valid key for chat functionality.")
        print("   But you can still ingest documents and test retrieval!")
    print("\nNext steps:")
    print("  1. Fix Gemini API key if needed")
    print("  2. Ingest documents: python main.py ingest <file>")
    print("  3. Test search manually with vector_store.py")
else:
    print("✗ SOME CORE COMPONENTS FAILED")
    print("Please fix the errors above before proceeding.")
    sys.exit(1)
