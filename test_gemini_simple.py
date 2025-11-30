import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {api_key[:20]}...")

try:
    genai.configure(api_key=api_key)
    print("API configured")
    
    # Try listing models
    print("\nAvailable models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
    
    # Try with gemini-pro
    print("\nTrying gemini-pro...")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Say OK')
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
