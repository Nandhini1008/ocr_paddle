import os
import sys
from dotenv import load_dotenv

def check_imports():
    print("Checking imports...")
    try:
        import paddle
        print("✅ paddlepaddle imported")
    except ImportError as e:
        print(f"❌ paddlepaddle failed: {e}")

    try:
        from paddleocr import PaddleOCR
        print("✅ paddleocr imported")
    except ImportError as e:
        print(f"❌ paddleocr failed: {e}")

    try:
        import qdrant_client
        print("✅ qdrant_client imported")
    except ImportError as e:
        print(f"❌ qdrant_client failed: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers imported")
    except ImportError as e:
        print(f"❌ sentence_transformers failed: {e}")

    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported")
    except ImportError as e:
        print(f"❌ google.generativeai failed: {e}")

def check_env():
    print("\nChecking .env...")
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        print("❌ GEMINI_API_KEY is missing or default in .env")
    else:
        print("✅ GEMINI_API_KEY found")
        
    host = os.getenv("QDRANT_HOST")
    print(f"ℹ️ QDRANT_HOST: {host}")

def check_poppler():
    print("\nChecking Poppler...")
    import shutil
    if shutil.which("pdftoppm"):
        print("✅ Poppler found (pdftoppm in PATH)")
    else:
        print("❌ Poppler NOT found in PATH. PDF to Image conversion will fail.")
        print("   (Please download Poppler for Windows and add 'bin' to PATH)")

def check_qdrant():
    print("\nChecking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        import config
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=2)
        collections = client.get_collections()
        print("✅ Qdrant connected")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   (Ensure Docker is running and Qdrant container is up)")

if __name__ == "__main__":
    check_imports()
    check_env()
    check_poppler()
    # check_qdrant() # Call this only if imports passed, but safe to call in try/except
    try:
        check_qdrant()
    except:
        pass
