import sys
print("Starting app.py...")
try:
    import gradio as gr
    print(f"Imported gradio version: {gr.__version__}")
    import os
    import shutil
    import docx
    print("Imported standard libs")
    
    print("Importing ocr_engine...")
    from ocr_engine import extract_text
    print("Imported ocr_engine")
    
    print("Importing vector_store...")
    from vector_store import VectorDB
    print("Imported vector_store")
    
    print("Importing chatbot...")
    from chatbot import RAGChatbot
    print("Imported chatbot")
except Exception as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize core components
try:
    vector_db = VectorDB()
    chatbot = RAGChatbot()
    print("System initialized successfully")
except Exception as e:
    print(f"Error initializing system: {e}")

def process_file(files):
    """
    Process uploaded files (PDF, Image, DOCX) and ingest into VectorDB
    """
    if not files:
        return "No files uploaded."
    
    status_msg = ""
    total_chunks = 0
    
    # Handle single file or list of files
    if not isinstance(files, list):
        files = [files]
    
    for file in files:
        try:
            # Gradio passes file objects or paths depending on version/config
            # In recent versions with file_count="multiple", it's a list of temp file paths or objects
            if hasattr(file, 'name'):
                file_path = file.name
            else:
                file_path = file
                
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            
            text = ""
            
            # 1. Extract Text based on file type
            if ext in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                status_msg += f"Processing {filename} with OCR...\n"
                text = extract_text(file_path)
                
            elif ext == '.docx':
                status_msg += f"Processing {filename} (Word Doc)...\n"
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                
            elif ext == '.txt':
                status_msg += f"Processing {filename} (Text)...\n"
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                status_msg += f"Skipping {filename}: Unsupported format {ext}\n"
                continue
                
            if not text.strip():
                status_msg += f"⚠️ No text extracted from {filename}\n"
                continue
                
            # 2. Ingest into VectorDB
            chunks = vector_db.add_document(text, metadata={'source': filename})
            total_chunks += chunks
            status_msg += f"✅ Successfully ingested {filename} ({chunks} chunks)\n"
            
        except Exception as e:
            status_msg += f"❌ Error processing {filename}: {str(e)}\n"
            
    return f"{status_msg}\nTotal new chunks added: {total_chunks}"

def chat_response(message, history):
    """
    Generate response using RAG Chatbot
    """
    if not message:
        return ""
        
    try:
        response = chatbot.query(message)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Define Theme - Simplified to avoid errors
try:
    theme = gr.themes.Soft()
except:
    theme = "default"

# Build the UI
with gr.Blocks(theme=theme, title="RAG Chatbot with OCR") as app:
    gr.Markdown(
        """
        # 🤖 RAG Chatbot with OCR
        Upload your documents (PDF, Images, DOCX) and ask questions about them!
        """
    )
    
    with gr.Row():
        # Left Column: File Upload
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Document Upload")
            file_upload = gr.File(
                file_count="multiple",
                label="Upload Documents",
                file_types=[".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
            )
            upload_btn = gr.Button("Process Documents", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=10)
            
            upload_btn.click(
                fn=process_file,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            
        # Right Column: Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot_ui = gr.ChatInterface(
                fn=chat_response,
                examples=[
                    "Summarize the uploaded documents.",
                    "What are the key points?",
                    "Explain the technical details."
                ],
                theme="soft"
            )

if __name__ == "__main__":
    print("Launching Gradio app...")
    app.launch(share=True)
