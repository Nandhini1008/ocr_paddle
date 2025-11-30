# OCR RAG Chatbot

This project implements a RAG (Retrieval-Augmented Generation) chatbot that extracts text from PDFs and Images using a custom OCR pipeline (CRAFT + PaddleOCR), stores the embeddings in Qdrant, and uses Google Gemini to answer questions.

## Prerequisites

1.  **Python 3.8+**
2.  **Docker Desktop** (Required for Qdrant)
3.  **Google Gemini API Key**

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - The `.env` file has been created from `.env.example`.
    - Open `.env` and set your `GEMINI_API_KEY`.
    - You can also adjust `LLM_MODEL_NAME` if needed (default: `gemini-1.5-flash`).

3.  **Start Qdrant**:
    Since Qdrant is self-hosted via Docker, you need to start it:
    ```bash
    docker-compose up -d
    ```
    *Note: Ensure Docker is running and `docker-compose` is in your PATH.*

## Usage

### 1. Ingest Documents
Extract text from a file and store it in the vector database.

```bash
python main.py ingest path/to/your/document.pdf
# OR
python main.py ingest path/to/your/image.jpg
```

### 2. Chat
Start the chatbot to query your documents.

```bash
python main.py chat
```

## Project Structure

- `main.py`: Entry point for CLI commands.
- `ocr_engine.py`: Handles text extraction using CRAFT and PaddleOCR.
- `vector_store.py`: Manages Qdrant connection and embedding generation (Sentence Transformers).
- `chatbot.py`: Connects the Vector Store with Gemini LLM for RAG.
- `config.py`: Configuration settings.
- `docker-compose.yml`: Docker service for Qdrant.
