"""
Demo: Test document ingestion and retrieval
"""
from vector_store import VectorDB

print("=" * 70)
print("DEMO: Document Ingestion & Semantic Search")
print("=" * 70)

# Initialize VectorDB
print("\n1. Initializing VectorDB...")
db = VectorDB()
print("   ✓ Connected to Qdrant")

# Ingest sample documents
print("\n2. Ingesting sample documents...")

doc1 = """
Python Programming Language
Python is a high-level, interpreted programming language.
It was created by Guido van Rossum and first released in 1991.
Python emphasizes code readability with significant whitespace.
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
"""

doc2 = """
Machine Learning Basics
Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.
Common algorithms include decision trees, neural networks, and support vector machines.
Machine learning is used in recommendation systems, image recognition, and natural language processing.
"""

doc3 = """
Web Development with Flask
Flask is a lightweight web framework for Python.
It is designed to make getting started quick and easy.
Flask provides tools, libraries, and technologies for building web applications.
It is often used for building RESTful APIs and microservices.
"""

num1 = db.add_document(doc1, metadata={'source': 'python_intro.txt', 'topic': 'programming'})
print(f"   ✓ Document 1: {num1} chunks (Python Programming)")

num2 = db.add_document(doc2, metadata={'source': 'ml_basics.txt', 'topic': 'ai'})
print(f"   ✓ Document 2: {num2} chunks (Machine Learning)")

num3 = db.add_document(doc3, metadata={'source': 'flask_guide.txt', 'topic': 'web'})
print(f"   ✓ Document 3: {num3} chunks (Flask Web Dev)")

# Test semantic search
print("\n3. Testing Semantic Search...")
print("\n" + "-" * 70)

queries = [
    "What is Python?",
    "Tell me about machine learning",
    "How to build web applications?",
    "Who created Python?"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = db.search(query, k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (Score: {result['score']:.4f}):")
        print(f"  Source: {result['metadata'].get('source', 'unknown')}")
        print(f"  Text: {result['text'][:100]}...")

print("\n" + "=" * 70)
print("✓ DEMO COMPLETE!")
print("\nThe RAG system successfully:")
print("  • Ingested 3 documents")
print("  • Created embeddings for each chunk")
print("  • Performed semantic search")
print("  • Retrieved relevant results based on meaning, not just keywords")
print("=" * 70)
