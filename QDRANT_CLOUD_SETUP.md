# Qdrant Cloud Setup Guide

## Step 1: Create Free Qdrant Cloud Account

1. Go to https://cloud.qdrant.io/
2. Sign up for a free account
3. Click "Create Cluster"
4. Select **FREE tier** (1GB storage, perfect for demo)
5. Choose a region close to you
6. Click "Create"

## Step 2: Get Your Credentials

After cluster is created:
1. Click on your cluster name
2. Copy the **Cluster URL** (looks like: `https://xxxxx-xxxxx.aws.cloud.qdrant.io:6333`)
3. Click "API Keys" → "Generate API Key"
4. Copy the API key

## Step 3: Update Your .env File

Add these to your `.env` file:

```bash
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here

# Keep these for local development
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Step 4: Update Streamlit Cloud Secrets

In Streamlit Cloud dashboard:
1. Go to your app settings
2. Click "Secrets"
3. Add:

```toml
GEMINI_API_KEY = "AIzaSyDmfa7jPM4IEOyZc-o83lTPgu2aXLg7vPQ"
LLM_MODEL_NAME = "gemini-2.0-flash-exp"

# Qdrant Cloud
QDRANT_URL = "https://your-cluster-url.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-qdrant-api-key"
```

## Step 5: Deploy!

Your app will automatically use Qdrant Cloud when deployed to Streamlit Cloud, and localhost when running locally.

---

## Quick Setup Commands

```bash
# 1. Update .env with Qdrant Cloud credentials
# 2. Test locally
streamlit run streamlit_app.py

# 3. Commit and push
git add .
git commit -m "Add Qdrant Cloud support"
git push origin main
```

Your Streamlit Cloud app will auto-redeploy with cloud Qdrant!
