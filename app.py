import streamlit as st
import os
import tempfile
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
        background: linear-gradient(135deg, #4F8BF9, #38BDF8, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        display: inline-block;
        margin-bottom: 4px;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<p class="main-header">⚛️ PhysRAG-Materials</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous RAG & Agentic Workflow for Solid-State Electronics, Memristors, Metrology & Quantum Information</p>', unsafe_allow_html=True)

st.markdown("""
<span class="badge">Solid State Physics</span>
<span class="badge">Quantum Information Theory</span>
<span class="badge">Groq Llama 3.3 (Free) / Gemini / Claude</span>
<span class="badge">ChromaDB Vector Search</span>
<span class="badge">DFT Code Generator</span>
<span class="badge">Neuromorphic Computing</span>
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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    
    # Knowledge Base Stats
    if engine:
        stats = engine.get_kb_stats()
        st.markdown("### 📊 Knowledge Base Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["total_files"])
        with col2:
            st.metric("Vector Chunks", stats["total_chunks"])
        st.caption(f"Total KB Size: {stats['total_size_kb']} KB")
    
    st.divider()
    st.markdown("### 📚 Project Metadata")
    st.markdown("""
    **Developer:** Abhijith Krishnan B M  
    **Institution:** IIST (Engineering Physics & Solid State Physics)  
    **Core Stack:**
    - Python & LangChain
    - ChromaDB (HuggingFace Embeddings)
    - Groq Llama 3.3 70B / Gemini / Claude
    - Streamlit Cloud Interface
    """)
    st.divider()
    st.markdown("[⭐ GitHub Repository](https://github.com/abhibm9602-cyber/Physrag-materials)")

# Main Tabbed Interface
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Literature RAG Search", "⚡ DFT & Code Generator", "📤 Upload Documents", "📖 Knowledge Base Index"])

# ---------------------------------------------------------
# TAB 1: RAG Literature Search
# ---------------------------------------------------------
with tab1:
    st.subheader("Query Physics Knowledge Base")
    
    sample_queries = [
        "Select a sample query...",
        # Memristors & DFT
        "Explain the switching mechanism in ITO/Al2O3/Au memristor heterostructures.",
        "What are the recommended DFT k-point grid and cutoff energy parameters for Al2O3?",
        "Compare PAW, ultrasoft, and norm-conserving pseudopotentials for DFT calculations.",
        # Metrology
        "How is Allan deviation calculated for rubidium atomic frequency standards?",
        # Neuromorphic
        "Explain the crossbar array architecture for neuromorphic computing.",
        # Quantum Information
        "What are Bell states and how are they used in quantum teleportation?",
        "What is the CHSH inequality and how does quantum mechanics violate it?",
        # QFT
        "Explain the path integral formulation of quantum field theory.",
        "What is spontaneous symmetry breaking and the Higgs mechanism?",
        # Condensed Matter
        "Derive the Fermi energy for a 3D free electron gas.",
        "Explain BCS theory of superconductivity and Cooper pair formation.",
        # ML for Materials
        "What are machine-learned interatomic potentials (MACE, NequIP, DeePMD)?",
        # Semiconductor Devices
        "Explain Fowler-Nordheim tunneling in ultra-thin oxide barriers.",
        # Statistical Mechanics
        "Derive the canonical partition function and its connection to free energy.",
        # MD Simulations
        "Explain the velocity-Verlet algorithm for molecular dynamics.",
        # Spectroscopy
        "How do you extract the band gap from a Tauc plot using UV-Vis spectroscopy?"
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
                
                # Performance metrics
                st.markdown("### ⏱️ Performance Metrics")
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{res.get('retrieval_time', 0):.3f}s</div><div class="metric-label">Retrieval Time</div></div>""", unsafe_allow_html=True)
                with mcol2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(res['retrieved_chunks'])}</div><div class="metric-label">Chunks Retrieved</div></div>""", unsafe_allow_html=True)
                with mcol3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-value">{res.get('generation_time', 0):.2f}s</div><div class="metric-label">AI Generation Time</div></div>""", unsafe_allow_html=True)
                
                st.write("")
                st.markdown("### 🤖 AI Synthesis Answer")
                st.markdown(res["answer"])
                
                st.divider()
                st.markdown("### 📑 Retrieved Reference Chunks")
                for i, chunk in enumerate(res["retrieved_chunks"]):
                    source_file = res["sources"][i] if i < len(res["sources"]) else "Unknown"
                    with st.expander(f"Reference Chunk {i+1} — {os.path.basename(source_file)}"):
                        st.write(chunk)
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "answer": res["answer"][:500] + "..." if len(res["answer"]) > 500 else res["answer"],
                    "chunks": len(res["retrieved_chunks"]),
                    "time": f"{res.get('retrieval_time', 0):.3f}s"
                })
    
    # Chat history
    if st.session_state.chat_history:
        st.divider()
        st.markdown("### 💬 Session History")
        for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {entry['query'][:80]}..."):
                st.write(entry["answer"])
                st.caption(f"Chunks: {entry['chunks']} | Retrieval: {entry['time']}")

# ---------------------------------------------------------
# TAB 2: DFT & Code Generator Agent
# ---------------------------------------------------------
with tab2:
    st.subheader("RAG-Augmented Physics Code Generator")
    st.caption("Now powered by knowledge base context — the AI retrieves relevant parameters before generating code!")
    
    col1, col2 = st.columns(2)
    with col1:
        code_category = st.selectbox("Code Category:", [
            "🔬 DFT & Electronic Structure",
            "🧪 Molecular Dynamics",
            "📊 Data Analysis & Metrology",
            "💻 Quantum Computing",
            "🧠 Machine Learning for Materials",
            "📐 Statistical Mechanics"
        ])
    
    # Dynamic target types based on category
    code_type_map = {
        "🔬 DFT & Electronic Structure": [
            "Quantum Espresso — SCF Calculation",
            "Quantum Espresso — Band Structure",
            "Quantum Espresso — DOS / PDOS",
            "Quantum Espresso — Phonon Calculation (DFPT)",
            "Quantum Espresso — Structural Relaxation",
            "VASP — INCAR/POSCAR/KPOINTS Generation",
            "Python — DFT Post-Processing (pymatgen)"
        ],
        "🧪 Molecular Dynamics": [
            "LAMMPS — Metal Simulation (EAM)",
            "LAMMPS — Thermal Conductivity (Green-Kubo)",
            "Python — MD with ASE (Atomic Simulation Environment)",
            "Python — Radial Distribution Function Analysis"
        ],
        "📊 Data Analysis & Metrology": [
            "Python — Allan Variance / Allan Deviation",
            "Python — Phase Noise Analysis",
            "Python — XRD Pattern Simulation",
            "Python — Tauc Plot Band Gap Extraction",
            "Python — IV Curve Analysis (Memristor)",
            "Python — Impedance Spectroscopy (Nyquist Plot)"
        ],
        "💻 Quantum Computing": [
            "Qiskit — Quantum Teleportation Circuit",
            "Qiskit — Grover's Search Algorithm",
            "Qiskit — VQE (Variational Quantum Eigensolver)",
            "Qiskit — Bell State Preparation & Measurement",
            "Cirq — Quantum Circuit Simulation"
        ],
        "🧠 Machine Learning for Materials": [
            "Python — Crystal Graph Neural Network (CGCNN)",
            "Python — Band Gap Prediction (Random Forest)",
            "Python — Formation Energy Prediction",
            "Python — Materials Project API Data Fetch",
            "Python — SOAP Descriptor Calculation"
        ],
        "📐 Statistical Mechanics": [
            "Python — 2D Ising Model (Monte Carlo)",
            "Python — Metropolis Algorithm Simulation",
            "Python — Fermi-Dirac Distribution Plotter",
            "Python — Phonon Density of States",
            "Python — Boltzmann Transport Equation Solver"
        ]
    }
    
    with col1:
        target_type = st.selectbox("Specific Script:", code_type_map.get(code_category, ["Quantum Espresso — SCF Calculation"]))
    with col2:
        system_input = st.text_input("Material/System Parameters:", "Al2O3 barrier 1.6 nm on ITO substrate, cutoff 40 Ry")
        
    if st.button("⚡ Generate Code Script"):
        if engine:
            with st.spinner("Agentic Workflow: Retrieving KB context, drafting, and critiquing script..."):
                result = engine.generate_dft_or_code_script(target_type, system_input)
                
                if isinstance(result, dict):
                    code_output = result["code"]
                    initial_code = result.get("initial_code", "")
                    critique = result.get("critique", "")
                    context_used = result.get("context_used", [])
                    
                    # Performance metrics
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.get('retrieval_time', 0):.3f}s</div><div class="metric-label">KB Retrieval</div></div>""", unsafe_allow_html=True)
                    with mcol2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{result.get('generation_time', 0):.2f}s</div><div class="metric-label">Dual-Agent Generation</div></div>""", unsafe_allow_html=True)
                    st.write("")
                    
                    if context_used:
                        with st.expander("📚 Knowledge Base Context Used for Generation"):
                            for i, ctx in enumerate(context_used[:3]):
                                st.info(f"**Context {i+1}:** {ctx[:200]}...")
                                
                    if initial_code:
                        with st.expander("🤖 Agent 1: Initial Draft Script"):
                            st.code(initial_code, language="python" if "Python" in target_type else "text")
                            
                    if critique:
                        st.warning(f"**🕵️ Agent 2 (Physics Critic):** {critique}")
                else:
                    code_output = result
                    
                st.markdown("### ✨ Final Refined Production Code:")
                if "Python" in target_type or "Qiskit" in target_type or "Cirq" in target_type:
                    lang = "python"
                elif "LAMMPS" in target_type:
                    lang = "bash"
                elif "VASP" in target_type:
                    lang = "text"
                else:
                    lang = "fortran"
                st.code(code_output, language=lang)

# ---------------------------------------------------------
# TAB 3: Upload Documents
# ---------------------------------------------------------
with tab3:
    st.subheader("📤 Upload New Research Documents")
    st.markdown("Drag and drop `.txt` or `.pdf` files to expand the knowledge base. The vector database will automatically reindex.")
    
    uploaded_files = st.file_uploader(
        "Upload research papers or reference documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Supported formats: .txt, .pdf"
    )
    
    if uploaded_files and st.button("📥 Index Uploaded Files", type="primary"):
        saved_paths = []
        for uploaded_file in uploaded_files:
            save_path = os.path.join("./data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(save_path)
            st.success(f"✅ Saved: `{uploaded_file.name}`")
        
        if saved_paths:
            with st.spinner("Reindexing ChromaDB with new documents..."):
                st.cache_resource.clear()
                engine_new = get_rag_engine()
                st.success(f"🎉 Successfully indexed {len(saved_paths)} new file(s)! Refresh the page to use the updated knowledge base.")

# ---------------------------------------------------------
# TAB 4: Knowledge Base Index Inspector
# ---------------------------------------------------------
with tab4:
    st.subheader("Knowledge Base Index Inspector")
    
    if engine:
        stats = engine.get_kb_stats()
        
        # Stats overview
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{stats['total_files']}</div><div class="metric-label">Total Documents</div></div>""", unsafe_allow_html=True)
        with scol2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{stats['total_chunks']}</div><div class="metric-label">Vector Chunks</div></div>""", unsafe_allow_html=True)
        with scol3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{stats['total_size_kb']} KB</div><div class="metric-label">Total KB Size</div></div>""", unsafe_allow_html=True)
        
        st.write("")
        st.markdown("### 📄 Indexed Files")
        for file_info in stats["file_details"]:
            st.info(f"📄 `{file_info['name']}` — {file_info['size_kb']} KB")
    else:
        st.warning("Engine not initialized.")
