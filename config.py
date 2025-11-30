import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant Configuration - supports both local and cloud
QDRANT_URL = os.getenv("QDRANT_URL")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # For local
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))  # For local
COLLECTION_NAME = "ocr_documents"

# Embedding Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # A good default sentence-transformer

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-exp")

# OCR Configuration
# Using the existing files for OCR
