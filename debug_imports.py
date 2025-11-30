import sys
print(f"Python {sys.version}")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"Transformers import failed: {e}")

try:
    import sentence_transformers
    print(f"Sentence Transformers version: {sentence_transformers.__version__}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")
except Exception as e:
    print(f"Sentence Transformers error: {e}")
    import traceback
    traceback.print_exc()
