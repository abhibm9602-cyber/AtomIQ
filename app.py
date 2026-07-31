import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import PhysRAGEngine

# Load env variables
load_dotenv(override=True)

# Page Configuration
st.set_page_config(
    page_title="PhysRAG-Materials | AI Agent for Physics & DFT",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4F8BF9;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AABF;
        margin-bottom: 25px;
    }
    .badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        border: 1px solid #0284C7;
        margin-right: 8px;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<p class="main-header">⚛️ PhysRAG-Materials</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous RAG & Agentic Workflow for Solid-State Electronics, Memristors, & Metrology</p>', unsafe_allow_html=True)

st.markdown("""
<span class="badge">Solid State Physics</span>
<span class="badge">Groq Llama 3.3 (100% Free) / Gemini / Claude</span>
<span class="badge">ChromaDB Vector Search</span>
<span class="badge">DFT Code Generator</span>
""", unsafe_allow_html=True)

st.write("")

# Initialize RAG Engine in Session State (Cached)
@st.cache_resource
def get_rag_engine():
    return PhysRAGEngine()

try:
    engine = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize PhysRAG Engine: {e}")
    engine = None

# Sidebar Configuration & Settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/physics.png", width=70)
    st.title("Settings & Control")
    
    saved_key = os.getenv("GROQ_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")))
    api_key_input = st.text_input("AI API Key (Groq / Gemini / Claude)", value=saved_key, type="password", help="Enter Groq API Key (gsk_...), Gemini API Key, or Claude API Key")
    
    if api_key_input and engine:
        if api_key_input.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = api_key_input
        elif api_key_input.startswith("AIzaSy") or api_key_input.startswith("AQ"):
            os.environ["GEMINI_API_KEY"] = api_key_input
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
            
        if hasattr(engine, 'update_llm'):
            engine.update_llm(api_key_input)
        
    if engine and engine.llm:
        st.success(f"AI Provider Active: {engine.active_provider}")
    else:
        st.info("Paste Free Groq API Key (gsk_...) above or add to .env")
        
    st.divider()
    st.markdown("### 📚 Project Metadata")
    st.markdown("""
    **Developer:** Abhijith Krishnan B M  
    **Institution:** IIST (Solid State Physics)  
    **Core Stack:**
    - Python & PyPDF
    - ChromaDB (HuggingFace Embeddings)
    - Groq Llama 3.3 70B / Gemini / Claude
    - Streamlit Cloud Interface
    """)
    st.divider()
    st.markdown("[⭐ GitHub Repository](https://github.com/abhibm9602-cyber)")

# Main Tabbed Interface
tab1, tab2, tab3 = st.tabs(["🔍 Literature RAG Search", "⚡ DFT & Code Generator", "📖 Knowledge Base Index"])

# ---------------------------------------------------------
# TAB 1: RAG Literature Search
# ---------------------------------------------------------
with tab1:
    st.subheader("Query Physics Knowledge Base & Memristor Research")
    
    sample_queries = [
        "Select a sample query...",
        "Explain the switching mechanism in ITO/Al2O3/Au memristor heterostructures.",
        "What are the recommended DFT k-point grid and cutoff energy parameters for Al2O3?",
        "How is Allan deviation calculated for rubidium atomic frequency standards?"
    ]
    
    selected_sample = st.selectbox("Sample Physics Queries:", sample_queries)
    default_query = selected_sample if selected_sample != "Select a sample query..." else ""
    
    user_query = st.text_area("Ask a scientific/technical question:", value=default_query, height=100, placeholder="e.g., Describe the dynamic filamentary switching in 1.6 nm tunneling barriers.")
    
    if st.button("🚀 Run RAG Query", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query first.")
        elif engine:
            with st.spinner("Searching ChromaDB vectors & generating AI response..."):
                res = engine.query_rag(user_query)
                
                st.markdown("### 🤖 AI Synthesis Answer")
                st.markdown(res["answer"])
                
                st.divider()
                st.markdown("### 📑 Retrieved Reference Chunks")
                for i, chunk in enumerate(res["retrieved_chunks"]):
                    with st.expander(f"Reference Chunk {i+1}"):
                        st.write(chunk)

# ---------------------------------------------------------
# TAB 2: DFT & Code Generator Agent
# ---------------------------------------------------------
with tab2:
    st.subheader("Automated Physics Code & DFT Script Generator")
    
    col1, col2 = st.columns(2)
    with col1:
        target_type = st.selectbox("Target Code Type:", ["Quantum Espresso (DFT Input)", "Python Metrology (Allan Variance Analysis)"])
    with col2:
        system_input = st.text_input("Material/System Parameters:", "Al2O3 barrier 1.6 nm on ITO substrate, cutoff 40 Ry")
        
    if st.button("⚡ Generate Code Script"):
        if engine:
            with st.spinner("Generating script via AI Engine..."):
                code_output = engine.generate_dft_or_code_script(target_type, system_input)
                st.markdown("### Generated Production Code:")
                st.code(code_output, language="python" if "Python" in target_type else "fortran")

# ---------------------------------------------------------
# TAB 3: Knowledge Base Index Inspector
# ---------------------------------------------------------
with tab3:
    st.subheader("Current Loaded Knowledge Base Files")
    data_files = [f for f in os.listdir("./data") if not f.startswith(".")]
    
    if data_files:
        for f in data_files:
            st.info(f"📄 `./data/{f}`")
    else:
        st.warning("No files currently in `./data/`. Place your `.pdf` or `.txt` research papers here to index them.")
