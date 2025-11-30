"""
Quick start script to verify the RAG Chatbot setup
"""
import os
import sys

def check_env_file():
    """Check if .env file is properly configured"""
    print("📋 Checking .env file...")
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        
    if 'your_api_key_here' in content:
        print("❌ GEMINI_API_KEY not configured in .env file")
        print("   Please update .env with your actual API key")
        return False
    
    print("✅ .env file configured")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    required = [
        'paddleocr',
        'qdrant_client',
        'sentence_transformers',
        'google.generativeai',
        'dotenv'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'google.generativeai':
                __import__('google.generativeai')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_qdrant():
    """Check if Qdrant is running"""
    print("\n🗄️  Checking Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print("✅ Qdrant is running")
        print(f"   Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Start Qdrant with: docker-compose up -d")
        return False

def check_gemini_api():
    """Check if Gemini API is accessible"""
    print("\n🤖 Checking Gemini API...")
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            print("❌ GEMINI_API_KEY not set properly")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello'")
        print("✅ Gemini API is working")
        print(f"   Response: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini API check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 RAG Chatbot Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Qdrant Database", check_qdrant),
        ("Gemini API", check_gemini_api)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to use the chatbot.")
        print("\n📝 Next steps:")
        print("   1. Ingest a document: python main.py ingest <file_path>")
        print("   2. Start chatting: python main.py chat")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n🔧 Quick fixes:")
        if not results.get("Environment File"):
            print("   • Update .env with your Gemini API key")
        if not results.get("Dependencies"):
            print("   • Run: pip install -r requirements.txt")
        if not results.get("Qdrant Database"):
            print("   • Start Docker Desktop")
            print("   • Run: docker-compose up -d")
        if not results.get("Gemini API"):
            print("   • Verify your Gemini API key is correct")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
