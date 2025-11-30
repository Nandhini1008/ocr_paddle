# RAG Chatbot - Quick Start Guide

## Current Status
✅ Gemini API Key configured  
✅ Dependencies installed  
⚠️ Qdrant needs to be started  

## Next Steps

### 1. Start Qdrant Database
**Make sure Docker Desktop is running**, then execute:
```bash
docker-compose up -d
```

Verify it's running:
```bash
docker ps
```

You should see a container named `qdrant` running on port 6333.

### 2. Test the Complete Setup
```bash
python test_quick.py
```

All 4 tests should pass.

### 3. Ingest Your First Document

To add a PDF or image to the knowledge base:
```bash
python main.py ingest path/to/your/document.pdf
```

Example:
```bash
python main.py ingest ocr_order.py
```

This will:
- Extract text using PaddleOCR
- Split into chunks
- Generate embeddings
- Store in Qdrant vector database

### 4. Start Chatting

Launch the interactive chatbot:
```bash
python main.py chat
```

Ask questions about your ingested documents!

## Example Workflow

```bash
# 1. Start Qdrant
docker-compose up -d

# 2. Ingest a document
python main.py ingest sample_document.pdf

# 3. Chat with your documents
python main.py chat
```

## Troubleshooting

### Qdrant Connection Error
- Ensure Docker Desktop is running
- Run: `docker-compose up -d`
- Check: `docker ps` to verify container is running

### API Key Error
- Verify `.env` file has your actual Gemini API key
- Key should start with `AIzaSy...`

### Import Errors
- Run: `pip install -r requirements.txt`

## Project Structure

```
ocr_paddle/
├── main.py              # Main entry point
├── ocr_engine.py        # OCR text extraction
├── vector_store.py      # Qdrant vector database
├── chatbot.py           # RAG chatbot logic
├── config.py            # Configuration
├── .env                 # Environment variables
└── requirements.txt     # Dependencies
```

## How It Works

1. **Document Ingestion**: OCR extracts text from PDFs/images
2. **Embedding**: Text is converted to vectors using Sentence Transformers
3. **Storage**: Vectors stored in Qdrant for fast similarity search
4. **Query**: User asks a question
5. **Retrieval**: Top-k similar chunks retrieved from Qdrant
6. **Generation**: Gemini generates answer based on retrieved context

## Commands Reference

| Command | Description |
|---------|-------------|
| `python main.py ingest <file>` | Add document to knowledge base |
| `python main.py chat` | Start interactive chat |
| `python test_quick.py` | Verify setup |
| `docker-compose up -d` | Start Qdrant |
| `docker-compose down` | Stop Qdrant |
