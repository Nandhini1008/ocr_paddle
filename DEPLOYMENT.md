# RAG Chatbot Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE & Easy)

1. **Push your code to GitHub** ✅ (Already done!)

2. **Deploy to Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Nandhini1008/ocr_paddle`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Add Secrets:**
   - In Streamlit Cloud dashboard, go to "Settings" → "Secrets"
   - Add your environment variables:
   ```toml
   GEMINI_API_KEY = "AIzaSyDmfa7jPM4IEOyZc-o83lTPgu2aXLg7vPQ"
   QDRANT_HOST = "localhost"
   QDRANT_PORT = "6333"
   LLM_MODEL_NAME = "gemini-2.0-flash-exp"
   ```

4. **Your app will be live at:**
   `https://nandhini1008-ocr-paddle-streamlit-app-xxxxx.streamlit.app`

---

## Option 2: Hugging Face Spaces (FREE)

1. Create account at https://huggingface.co/
2. Create new Space → Select "Streamlit"
3. Upload your files or connect GitHub repo
4. Add secrets in Space settings
5. Your app will be at: `https://huggingface.co/spaces/YOUR_USERNAME/ocr-paddle`

---

## Option 3: Docker Local Deployment

### Build and Run:
```bash
# Build Docker image
docker build -t rag-chatbot .

# Run with Qdrant
docker-compose up -d

# Access at http://localhost:8501
```

### Docker Compose (Full Stack):
```bash
docker-compose up -d
```

This starts:
- Qdrant on port 6333
- Streamlit app on port 8501

---

## Option 4: Railway.app (FREE Tier Available)

1. Go to https://railway.app/
2. Sign in with GitHub
3. "New Project" → "Deploy from GitHub repo"
4. Select `Nandhini1008/ocr_paddle`
5. Add environment variables
6. Railway will auto-detect Dockerfile and deploy
7. Get public URL from Railway dashboard

---

## Recommended: Streamlit Cloud

**Pros:**
- ✅ Completely FREE
- ✅ Easy GitHub integration
- ✅ Auto-deploys on git push
- ✅ Built-in secrets management
- ✅ Public shareable link

**Steps:**
1. Visit https://share.streamlit.io/
2. Sign in with GitHub
3. Deploy `Nandhini1008/ocr_paddle`
4. Add API key in secrets
5. Share the generated link!

**Note:** For Qdrant, you'll need to either:
- Use Qdrant Cloud (free tier): https://cloud.qdrant.io/
- Or modify the app to use in-memory vector store for demo purposes
