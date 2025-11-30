import argparse
import os
import sys
from ocr_engine import extract_text
from vector_store import VectorDB
from chatbot import RAGChatbot

def ingest_file(file_path):
    print(f"Processing {file_path}...")
    try:
        text = extract_text(file_path)
        if not text:
            print("No text extracted.")
            return
            
        print(f"Extracted {len(text)} characters.")
        
        db = VectorDB()
        num_chunks = db.add_document(text, metadata={'source': os.path.basename(file_path)})
        print(f"Successfully indexed {num_chunks} chunks into Qdrant.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

def start_chat():
    print("Initializing Chatbot...")
    try:
        bot = RAGChatbot()
        print("Chatbot ready! Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = bot.query(user_input)
            print(f"Bot: {response}")
            
    except Exception as e:
        print(f"Error initializing chatbot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR RAG Chatbot")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a PDF or Image file')
    ingest_parser.add_argument('file_path', help='Path to the file')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start the chatbot')
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        ingest_file(args.file_path)
    elif args.command == 'chat':
        start_chat()
    else:
        parser.print_help()
