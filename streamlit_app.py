import streamlit as st
import os
import tempfile
from ocr_engine import extract_text
from vector_store import VectorDB
from chatbot import RAGChatbot
import docx

# Page config
st.set_page_config(page_title="RAG Chatbot with OCR", page_icon="🤖", layout="wide")

# Initialize resources with caching
@st.cache_resource
def get_vector_db():
    return VectorDB()

@st.cache_resource
def get_chatbot(_vector_db):
    return RAGChatbot(vector_db=_vector_db)

@st.cache_resource
def get_ocr_components():
    from ocr_pdf_pipeline import CRAFTDetector
    from paddleocr import PaddleOCR
    detector = CRAFTDetector()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_batch_num=1)
    return detector, ocr

# Initialize session state for components
if 'vector_db' not in st.session_state:
    try:
        with st.spinner("Initializing system... (this may take a minute)"):
            st.session_state.vector_db = get_vector_db()
            st.session_state.chatbot = get_chatbot(st.session_state.vector_db)
            st.session_state.detector, st.session_state.ocr = get_ocr_components()
            st.session_state.system_ready = True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.session_state.system_ready = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file"""
    try:
        # Save to temp file
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        filename = uploaded_file.name
        ext = suffix.lower()
        text = ""

        # Extract text
        if ext in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            with st.spinner(f"Running OCR on {filename}..."):
                text = extract_text(
                    tmp_path, 
                    detector=st.session_state.get('detector'), 
                    ocr=st.session_state.get('ocr')
                )
        elif ext == '.docx':
            doc = docx.Document(tmp_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == '.txt':
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Clean up temp file
        os.unlink(tmp_path)

        if not text.strip():
            return False, f"No text extracted from {filename}"

        # Ingest
        chunks = st.session_state.vector_db.add_document(text, metadata={'source': filename})
        return True, f"Ingested {filename} ({chunks} chunks)"

    except Exception as e:
        return False, f"Error processing {filename}: {str(e)}"

# Sidebar for file upload
with st.sidebar:
    st.title("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF, Images, DOCX, TXT", 
        accept_multiple_files=True,
        type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt']
    )
    
    if uploaded_files and st.button("Process Documents"):
        if not st.session_state.get('system_ready', False):
            st.error("System not initialized properly.")
        else:
            progress_bar = st.progress(0)
            status_area = st.empty()
            
            for i, file in enumerate(uploaded_files):
                success, msg = process_uploaded_file(file)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_area.success("Processing complete!")
            
    st.markdown("---")
    if st.button("⚠️ Reset Database"):
        try:
            from qdrant_client.http import models
            # Re-create the collection (this clears all data)
            st.session_state.vector_db.client.recreate_collection(
                collection_name=st.session_state.vector_db.collection_name,
                vectors_config=models.VectorParams(
                    size=st.session_state.vector_db.encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            st.success("Database cleared! Please re-upload your documents.")
        except Exception as e:
            st.error(f"Error resetting database: {e}")

# Main Chat Interface
st.title("🤖 RAG Chatbot with OCR")
st.markdown("Ask questions about your uploaded documents.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if st.session_state.get('system_ready', False):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.error("System is not ready. Please check logs.")
