
import sys
from vector_store import VectorDB

def test_mock_ingest():
    print("Testing Mock Ingestion...")
    try:
        db = VectorDB()
        print("VectorDB initialized.")
        
        text = "This is a test document. It contains some information about the RAG chatbot. The chatbot uses PaddleOCR and Qdrant."
        print(f"Ingesting text: {text}")
        
        num = db.add_document(text, metadata={'source': 'mock_test'})
        print(f"Ingested {num} chunks.")
        
        print("Testing Search...")
        results = db.search("What does the chatbot use?")
        for r in results:
            print(f"Found: {r['text']} (Score: {r['score']})")
            
        print("Mock Ingestion Test PASSED")
    except Exception as e:
        print(f"Mock Ingestion Test FAILED: {e}")

if __name__ == "__main__":
    test_mock_ingest()
