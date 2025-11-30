import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
print(f"Testing with API key: {api_key[:20]}...")

genai.configure(api_key=api_key)

# Try different model names
models_to_try = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro',
    'models/gemini-1.5-flash',
    'models/gemini-pro'
]

for model_name in models_to_try:
    try:
        print(f"\nTrying: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content('Say "OK"')
        print(f"  ✓ SUCCESS! Response: {response.text}")
        print(f"\n✓ Working model: {model_name}")
        break
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:80]}")
