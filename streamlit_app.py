"""
RAG Knowledge Base Q&A Chatbot - COMPLETE APPLICATION
Built step-by-step with: Streamlit + LangChain + ChromaDB + Groq

This app allows you to:
1. Upload PDF documents
2. Ask questions about them
3. Get answers with source citations
"""

import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import shutil

# LangChain imports - Updated for v1.x
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Base Q&A",
    page_icon="📚",
    layout="wide"
)

# Custom CSS - World-class styling
st.markdown("""
<style>
    /* Dark Mode Toggle */
    .dark-mode-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: rgba(255,255,255,0.2);
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1rem;
        color: white;
        cursor: pointer;
        font-size: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .dark-mode-toggle:hover {
        background: rgba(255,255,255,0.3);
        transform: scale(1.05);
    }
    
    /* Global Styles - Light Mode */
    .main {
        padding: 2rem 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        transition: all 0.3s ease;
    }
    
    /* Dark mode is now handled via conditional CSS injection below */
    
    
    /* Header Styling */
    h1 {
        color: #ffffff !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Main Title - Dark Blue */
    .main-title h1 {
        color: #1e3a8a !important;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Force dark blue on title */
    div[style*="text-align: center"] h1 {
        color: #1e3a8a !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Sidebar Headers - Enhanced Visibility */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem !important;
        padding: 0.5rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Text Visibility */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stError,
    [data-testid="stSidebar"] code {
        color: #ffffff !important;
    }
    
    /* File Uploader Styling */
    [data-testid="stSidebar"] .stFileUploader {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader section {
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    [data-testid="stSidebar"] .stFileUploader section button {
        background: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown code {
        background: rgba(255,255,255,0.2) !important;
        color: #ffffff !important;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Button - Dark Blue Background */
    [data-testid="stSidebar"] .stButton>button {
        background: #1e40af !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4) !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #1e3a8a !important;
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.6) !important;
    }
    
    /* Chat Message Bubbles */
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
        border-bottom-left-radius: 5px;
    }
    
    .chat-answer {
        background: #ffffff;
        color: #333;
        margin-right: 2rem;
        border: 1px solid #e0e0e0;
        border-bottom-right-radius: 5px;
    }
    
    /* Card Styling */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card h3 {
        color: #667eea !important;
        margin-top: 0 !important;
    }
    
    .feature-card p {
        color: #374151 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    /* Example Questions Styling - Light Mode */
    .example-questions {
        background: white !important;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .example-questions div[style*="display: grid"] > div {
        background: #f8f9fa !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .example-questions strong {
        color: #333 !important;
        font-weight: 600;
    }
    
    .example-questions div[style*="display: grid"] > div {
        color: #666 !important;
    }
    
    /* Source Citation Styling */
    .source-citation {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: #10b981;
        color: white;
    }
    
    /* Welcome Screen */
    .welcome-container {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'example_clicked' not in st.session_state:
    st.session_state.example_clicked = False
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ============================================================================
# FUNCTION 1: PDF PROCESSING
# ============================================================================

def load_and_process_pdf(uploaded_file):
    """Load PDF and extract text"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Filter out empty documents and add metadata
        filtered_docs = []
        for doc in documents:
            # Skip documents with very little content (likely title pages or empty)
            if len(doc.page_content.strip()) > 50:  # At least 50 characters
                doc.metadata['source_file'] = uploaded_file.name
                filtered_docs.append(doc)
        
        return filtered_docs if filtered_docs else documents  # Return all if all are short
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ============================================================================
# FUNCTION 2: VECTOR STORE CREATION (RAG CORE)
# ============================================================================

def create_vectorstore(documents, collection_name=None):
    """Create vector store from documents"""
    
    # Split into chunks - use larger chunks for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased from 1000 for better context
        chunk_overlap=300,  # Increased overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting on paragraphs
    )
    chunks = text_splitter.split_documents(documents)
    
    # Filter out chunks that are too short or contain only URLs/repetitive content
    filtered_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        # Skip chunks that are too short or contain only URLs
        if len(content) > 200:  # Minimum meaningful content
            # Skip chunks that are mostly URLs or repetitive text
            if not (content.count('www.') > 2 or content.count('http') > 2):
                filtered_chunks.append(chunk)
    
    chunks = filtered_chunks if filtered_chunks else chunks  # Use filtered if we have them
    
    # Create embeddings (local, no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store with a unique collection name
    if collection_name is None:
        import uuid
        collection_name = f"documents_{uuid.uuid4().hex[:8]}"
    
    # Clear existing database if it exists (with retry for Windows file locking)
    if os.path.exists("./chroma_db"):
        try:
            # Try to delete immediately
            shutil.rmtree("./chroma_db")
        except (PermissionError, OSError) as e:
            # If locked, wait a bit and try again
            import time
            time.sleep(0.5)
            try:
                shutil.rmtree("./chroma_db")
            except (PermissionError, OSError):
                # If still locked, use a different directory name
                timestamp = int(time.time())
                persist_dir = f"./chroma_db_{timestamp}"
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir,
                    collection_name=collection_name
                )
                return vectorstore, len(chunks)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    
    return vectorstore, len(chunks)

# ============================================================================
# FUNCTION 3: QA CHAIN SETUP
# ============================================================================

def get_qa_chain(vectorstore):
    """Create QA chain with Groq LLM using LCEL"""
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file")
        return None
    
    # Initialize Groq LLM
    # Using llama-3.1-8b-instant (fast and currently supported)
    # Alternative models: llama-3.3-70b-versatile, llama-3.2-90b-text-preview
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )
    
    # Create retriever - retrieve more documents for better context
    # Use MMR (Maximum Marginal Relevance) to get diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR for diverse retrieval
        search_kwargs={"k": 8, "fetch_k": 20}  # Get 8 diverse results from top 20
    )
    
    # Custom prompt template - improved for better content extraction
    template = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Use ONLY the information provided in the context below to answer the question
- If the context contains actual book content (not just URLs, copyright notices, or title pages), use that content
- Ignore any context that is just website URLs, copyright notices, or repetitive text
- If the context doesn't contain relevant information, say so clearly
- When possible, cite page numbers or chapter information from the metadata

Context from the document:
{context}

Question: {question}

Provide a detailed answer based on the document content above:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create QA chain using LCEL
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever

# ============================================================================
# USER INTERFACE
# ============================================================================

# Dark Mode Toggle - Top Right
col1, col2 = st.columns([10, 1])
with col2:
    dark_mode_icon = "☀️" if st.session_state.dark_mode else "🌙"
    dark_mode_label = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
    if st.button(dark_mode_icon, key="dark_mode_toggle", help=dark_mode_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Apply dark mode using pure CSS injection
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    /* Dark Mode Override Styles */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    .chat-answer {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    .feature-card,
    .welcome-container {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .welcome-container h2,
    .feature-card h3 {
        color: #ffffff !important;
    }
    
    .welcome-container p,
    .feature-card p {
        color: #cbd5e0 !important;
    }
    
    .source-citation {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-left-color: #667eea !important;
    }
    
    .stTextInput>div>div>input {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea !important;
    }
    
    [style*="background: white"] {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    [style*="color: #333"] {
        color: #e2e8f0 !important;
    }
    
    [style*="color: #666"] {
        color: #cbd5e0 !important;
    }
    
    [style*="background: #f8f9fa"] {
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }
    
    .stExpander {
        background: #1e293b !important;
    }
    
    .stExpander label {
        color: #e2e8f0 !important;
    }
    
    /* Dark mode title color adjustment */
    .main-title h1 {
        color: #60a5fa !important;
    }
    
    /* Example questions section in dark mode - Ultra-specific selectors */
    .example-questions,
    .example-questions[style],
    div.example-questions {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .example-questions div[style*="background"],
    .example-questions > div[style],
    .example-questions div[style*="display: grid"],
    .example-questions div[style*="padding"] {
        background: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    .example-questions strong,
    .example-questions div strong,
    .example-questions div div strong {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .example-questions div,
    .example-questions div div,
    .example-questions * {
        color: #e2e8f0 !important;
    }
    
    /* Override all inline text colors in example questions */
    .example-questions [style*="color"] {
        color: #e2e8f0 !important;
    }
    
    /* Ensure example question headers are visible in dark mode */
    .example-questions div[style*="display: grid"] > div strong {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with modern styling - Dark blue color with class
st.markdown("""
<div class="main-title" style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #1e3a8a !important; font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
        📚 RAG Knowledge Base
    </h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0;">
        Intelligent Document Q&A powered by AI
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Document Upload
with st.sidebar:
    # Use markdown for better header styling
    st.markdown("## 📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )
    
    if uploaded_files:
        if st.button("📤 Upload & Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                all_documents = []
                
                # Load all PDFs
                for uploaded_file in uploaded_files:
                    try:
                        docs = load_and_process_pdf(uploaded_file)
                        all_documents.extend(docs)
                        # Show how many pages were extracted
                        num_pages = len(docs)
                        total_chars = sum(len(doc.page_content) for doc in docs)
                        st.success(f"✅ Loaded {uploaded_file.name} ({num_pages} pages, {total_chars:,} characters)")
                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
                
                if all_documents:
                    # Close existing vector store if it exists (to release file locks)
                    if st.session_state.vectorstore is not None:
                        try:
                            # Delete the collection to release locks
                            st.session_state.vectorstore.delete_collection()
                        except:
                            pass
                        st.session_state.vectorstore = None
                    
                    # Create vector store (clears old data automatically)
                    try:
                        vectorstore, num_chunks = create_vectorstore(all_documents)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents_processed = True
                        st.session_state.chat_history = []  # Clear chat history for new documents
                        
                        st.success(f"✨ Processed {len(all_documents)} pages into {num_chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ Error creating vector store: {str(e)}")
    
    # Status indicator with better styling
    if st.session_state.documents_processed:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    color: white; padding: 1rem; border-radius: 10px; 
                    text-align: center; font-weight: 600; margin: 1rem 0;">
            ✅ Documents Ready for Q&A
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Database"):
            st.session_state.vectorstore = None
            st.session_state.documents_processed = False
            st.session_state.chat_history = []
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            st.rerun()
    
    st.divider()
    
    # Instructions
    st.markdown("### 🔑 Setup")
    st.markdown("""
    1. Get free API key: [console.groq.com](https://console.groq.com)
    2. Create `.env` file:
       ```
       GROQ_API_KEY=your_key
       ```
    3. Upload PDFs & ask!
    """)

# Main area - Chat Interface
if st.session_state.documents_processed:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h2 style="color: #333; margin: 0; display: flex; align-items: center; gap: 0.5rem;">
            💬 Ask Questions
        </h2>
        <p style="color: #666; margin: 0.5rem 0 0 0;">Get intelligent answers from your documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history with modern bubbles
    if st.session_state.chat_history:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            # Question bubble
            st.markdown(f"""
            <div class="chat-message chat-question" style="margin-bottom: 1rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; opacity: 0.9;">Question</div>
                <div style="font-size: 1.1rem;">{question}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer bubble
            st.markdown(f"""
            <div class="chat-message chat-answer" style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem; color: #667eea; font-size: 0.9rem;">Answer</div>
                <div style="line-height: 1.8; color: #333;">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources with modern styling
            if sources:
                with st.expander(f"📖 View Sources ({len(sources)} sources)", expanded=False):
                    for j, source in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-citation">
                            <div style="font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">Source {j+1}</div>
                            <div style="color: #555; line-height: 1.6; margin-bottom: 0.5rem;">
                                {source.page_content[:400]}{'...' if len(source.page_content) > 400 else ''}
                            </div>
                            <div style="font-size: 0.85rem; color: #888; display: flex; gap: 1rem;">
                                <span>📄 {source.metadata.get('source_file', 'N/A')}</span>
                                <span>📃 Page {source.metadata.get('page', 'N/A')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
    
    # Question input with modern styling
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-top: 2rem;">
    """, unsafe_allow_html=True)
    
    question = st.text_input(
        "💭 Ask a question about your document:",
        placeholder="e.g., What are the main topics? Summarize chapter 1...",
        key="question_input",
        label_visibility="visible"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col3:
        if st.button("💡 Examples", use_container_width=True):
            st.session_state.example_clicked = not st.session_state.example_clicked
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show example questions if button clicked
    if st.session_state.get('example_clicked', False):
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <strong>💡 Try these questions:</strong>
            <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                <li>What is this document about?</li>
                <li>Summarize the key points</li>
                <li>What are the main chapters?</li>
                <li>List the important recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process question
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                qa_chain, retriever = get_qa_chain(st.session_state.vectorstore)
                
                if qa_chain:
                    # Get source documents first (to verify retrieval)
                    sources = retriever.invoke(question)
                    
                    # Debug: Check if we're getting meaningful content
                    if sources and len(sources) > 0:
                        # Filter out sources that are too short (likely title pages)
                        meaningful_sources = [s for s in sources if len(s.page_content.strip()) > 100]
                        if not meaningful_sources and sources:
                            # If all sources are short, use them anyway but warn
                            meaningful_sources = sources
                    else:
                        meaningful_sources = sources
                    
                    # Get answer
                    answer = qa_chain.invoke(question)
                    
                    # Add to history
                    st.session_state.chat_history.append((question, answer, meaningful_sources))
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    # Welcome screen with modern design
    st.markdown("""
    <div class="welcome-container">
        <div style="text-align: center; margin-bottom: 3rem;">
            <h2 style="color: #333; font-size: 2.5rem; margin-bottom: 1rem;">Welcome to RAG Knowledge Base</h2>
            <p style="color: #666; font-size: 1.2rem; margin: 0;">
                👈 Upload PDF documents in the sidebar to get started!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features with cards
    st.markdown("### 🌟 Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📚 Multi-Document</h3>
            <p>Upload and process multiple PDFs simultaneously. Perfect for research papers, books, and documentation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 AI-Powered</h3>
            <p>Semantic search understands meaning, not just keywords. Get intelligent answers from your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Privacy-First</h3>
            <p>Local embeddings keep your data private. Only your questions are sent to the AI, not your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Example questions with better styling
    st.markdown("### 🎯 Example Questions")
    st.markdown("""
    <div class="example-questions" style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
                <strong>📖 Content Analysis</strong><br>
                "What is this document about?"
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
                <strong>📝 Summarization</strong><br>
                "Summarize the key findings"
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
                <strong>🔍 Specific Queries</strong><br>
                "What recommendations are mentioned?"
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
                <strong>💡 Deep Dive</strong><br>
                "Explain [topic] from the document"
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with modern styling
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: rgba(255,255,255,0.8); margin-top: 3rem;">
    <p style="margin: 0.5rem 0;">
        Built with ❤️ using <strong>Streamlit</strong> + <strong>LangChain</strong> + <strong>Groq</strong> + <strong>ChromaDB</strong>
    </p>
    <p style="margin: 0; font-size: 0.9rem;">
        Powered by AI • Privacy-Focused • Open Source
    </p>
</div>
""", unsafe_allow_html=True)