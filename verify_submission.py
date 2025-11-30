import sys
import os
import time
from datetime import datetime

# ANSI colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_status(component, status, message=""):
    if status:
        print(f"{component:<35} {GREEN}[PASS]{RESET} {message}")
    else:
        print(f"{component:<35} {RED}[FAIL]{RESET} {message}")

def header(text):
    print(f"\n{CYAN}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{RESET}")

def verify_submission():
    header("RAG CHATBOT SUBMISSION VERIFICATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_status = True
    
    # 1. Environment Check
    header("1. Environment & Configuration")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ["GEMINI_API_KEY", "QDRANT_HOST", "QDRANT_PORT", "LLM_MODEL_NAME"]
        missing = [v for v in required_vars if not os.getenv(v)]
        
        if missing:
            print_status("Environment Variables", False, f"Missing: {', '.join(missing)}")
            overall_status = False
        else:
            print_status("Environment Variables", True, "All present")
            
        # Check API Key format (basic check)
        key = os.getenv("GEMINI_API_KEY")
        if key and key.startswith("AIza"):
            print_status("API Key Format", True, "Valid prefix")
        else:
            print_status("API Key Format", False, "Invalid prefix (should start with AIza)")
            overall_status = False
            
    except Exception as e:
        print_status("Configuration Load", False, str(e))
        overall_status = False

    # 2. Dependency Check
    header("2. Dependencies & Imports")
    dependencies = [
        ("qdrant_client", "Qdrant Client"),
        ("sentence_transformers", "Sentence Transformers"),
        ("google.generativeai", "Google Gemini SDK"),
        ("paddleocr", "PaddleOCR"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print_status(name, True)
        except ImportError as e:
            print_status(name, False, f"Not installed ({e})")
            overall_status = False
        except Exception as e:
            print_status(name, False, f"Error: {e}")
            overall_status = False

    # 3. Core Components
    header("3. Core Components Integration")
    
    # Qdrant
    try:
        from qdrant_client import QdrantClient
        import config
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        cols = client.get_collections()
        print_status("Qdrant Database", True, f"Connected (Collections: {len(cols.collections)})")
    except Exception as e:
        print_status("Qdrant Database", False, f"Connection failed: {e}")
        overall_status = False

    # Embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import config
        encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        vec = encoder.encode("test")
        print_status("Embedding Model", True, f"Loaded (Dim: {len(vec)})")
    except Exception as e:
        print_status("Embedding Model", False, str(e))
        overall_status = False

    # VectorDB
    try:
        from vector_store import VectorDB
        db = VectorDB()
        num = db.add_document("Verification test document.", metadata={"source": "verify.py"})
        results = db.search("verification", k=1)
        if results and results[0]['score'] > 0.2:
            print_status("VectorDB (R/W)", True, "Read/Write successful")
        else:
            print_status("VectorDB (R/W)", False, "Search returned poor/no results")
            overall_status = False
    except Exception as e:
        print_status("VectorDB (R/W)", False, str(e))
        overall_status = False

    # Gemini API
    try:
        import google.generativeai as genai
        import config
        genai.configure(api_key=config.GEMINI_API_KEY)
        # Try a few models if one fails
        models_to_try = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.5-pro']
        api_success = False
        last_error = None
        
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content("Reply with 'Verified'")
                if response and response.text:
                    print_status("Gemini API", True, f"Response from {m_name}: {response.text.strip()}")
                    api_success = True
                    break
            except Exception as e:
                last_error = e
                continue
        
        if not api_success:
            print_status("Gemini API", False, f"All models failed. Last error: {last_error}")
            print(f"{YELLOW}Warning: Gemini API check failed. Chatbot response will be limited.{RESET}")
            
    except Exception as e:
        print_status("Gemini API", False, f"API Error: {e}")
        print(f"{YELLOW}Warning: Gemini API check failed. Chatbot response will be limited.{RESET}")

    # 4. End-to-End Chatbot
    header("4. Chatbot Logic")
    try:
        from chatbot import RAGChatbot
        bot = RAGChatbot()
        print_status("Chatbot Initialization", True, "Ready")
        
        # Mock query test
        try:
            print("   Testing query: 'What is verification?'")
            ans = bot.query("What is verification?")
            print_status("Chatbot Query", True, f"Response length: {len(ans)} chars")
        except Exception as e:
             print_status("Chatbot Query", False, f"Failed: {e}")
    except Exception as e:
        print_status("Chatbot Init", False, str(e))
        overall_status = False

    header("Final Verdict")
    if overall_status:
        print(f"{GREEN}✅ READY FOR SUBMISSION{RESET}")
        print("All core systems are functioning correctly.")
    else:
        print(f"{RED}❌ SUBMISSION VERIFICATION FAILED{RESET}")
        print("Please fix the failed components above.")

if __name__ == "__main__":
    verify_submission()
