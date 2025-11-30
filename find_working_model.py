import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

candidates = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-pro",
    "gemini-1.0-pro",
    "models/gemini-1.5-flash",
    "models/gemini-pro"
]

print("Testing models...")
working_model = None

for name in candidates:
    try:
        print(f"Trying {name}...", end=" ")
        model = genai.GenerativeModel(name)
        response = model.generate_content("Hello")
        print("SUCCESS!")
        working_model = name
        break
    except Exception as e:
        print(f"Failed ({str(e)[:50]}...)")

if working_model:
    print(f"\nFound working model: {working_model}")
    # Update .env
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('LLM_MODEL_NAME='):
                f.write(f'LLM_MODEL_NAME={working_model}\n')
            else:
                f.write(line)
    print("Updated .env with working model.")
else:
    print("\nNo working model found in candidates.")
