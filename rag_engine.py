import os
import glob
import time
import hashlib
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# LangChain Core & Integration Imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv(override=True)

class PhysRAGEngine:
    """
    RAG & Agentic Engine for Solid-State Physics, DFT Simulations, 
    Precision Metrology, Neuromorphic Computing, and Quantum Information Theory.
    
    Integrates ChromaDB vector store with multi-provider LLM support:
    - Groq AI (Llama 3.3 70B - 100% Free Unrestricted API Tier)
    - Google Gemini
    - Anthropic Claude 3.5 Sonnet
    
    Features:
    - Smart auto-reindexing when new files are added
    - RAG-augmented code generation
    - Performance metrics tracking
    - Optimized chunking (800 chars, 200 overlap)
    """
    INDEX_TRACKER = ".indexed_files.json"
    
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.last_retrieval_time = 0.0
        self.last_chunks_searched = 0
        self.total_chunks = 0
        
        # Initialize light local embedding model (no API key needed for vectorization!)
        print("[PhysRAG] Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load and Index Documents (with smart reindexing)
        self.vectorstore = self._build_or_load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Setup LLM dynamically
        self.update_llm()

    def _get_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of a file for change detection."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_indexed_files(self) -> Dict[str, str]:
        """Load the index tracker (filename -> hash mapping)."""
        tracker_path = os.path.join(self.persist_dir, self.INDEX_TRACKER)
        if os.path.exists(tracker_path):
            try:
                with open(tracker_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_indexed_files(self, indexed: Dict[str, str]):
        """Save the index tracker."""
        os.makedirs(self.persist_dir, exist_ok=True)
        tracker_path = os.path.join(self.persist_dir, self.INDEX_TRACKER)
        with open(tracker_path, 'w') as f:
            json.dump(indexed, f, indent=2)

    def _discover_files(self) -> List[str]:
        """Discover all indexable files in the data directory."""
        files = []
        for ext in ["*.txt", "*.pdf"]:
            files.extend(glob.glob(os.path.join(self.data_dir, ext)))
        return sorted(files)

    def update_llm(self, api_key: str = None):
        """Updates or re-initializes LLM dynamically (Groq Free vs Gemini vs Claude)."""
        load_dotenv(override=True)
        
        groq_key = api_key if (api_key and api_key.startswith("gsk_")) else os.getenv("GROQ_API_KEY")
        gemini_key = api_key if (api_key and (api_key.startswith("AIzaSy") or api_key.startswith("AQ"))) else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        anthropic_key = api_key if (api_key and api_key.startswith("sk-ant")) else os.getenv("ANTHROPIC_API_KEY")
        
        self.llm = None
        self.active_provider = "None"
        
        # Priority 1: Groq AI (Llama 3.3 70B - 100% Free & Fast)
        if groq_key and groq_key.strip():
            try:
                self.llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=groq_key.strip(),
                    temperature=0.2
                )
                self.active_provider = "Groq Llama 3.3 70B (100% Free)"
                return
            except Exception as e:
                print(f"[PhysRAG Warning] Failed with Groq: {e}")

        # Priority 2: Google Gemini
        if gemini_key and gemini_key.strip():
            for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        google_api_key=gemini_key.strip(),
                        temperature=0.2
                    )
                    self.active_provider = f"Google Gemini ({model_name})"
                    return
                except Exception as e:
                    print(f"[PhysRAG Warning] Failed with Gemini {model_name}: {e}")
                    
        # Priority 3: Anthropic Claude 3.5
        if anthropic_key and anthropic_key.strip():
            try:
                self.llm = ChatAnthropic(
                    model_name="claude-3-5-sonnet-20240620",
                    anthropic_api_key=anthropic_key.strip(),
                    temperature=0.2
                )
                self.active_provider = "Claude 3.5 Sonnet"
                return
            except Exception as e:
                print(f"[PhysRAG Warning] Failed to initialize Anthropic LLM: {e}")

    def _build_or_load_vectorstore(self) -> Chroma:
        """Loads documents and creates/updates Chroma vector DB with smart reindexing."""
        all_files = self._discover_files()
        indexed_files = self._get_indexed_files()
        
        # Check which files are new or modified
        new_or_modified = []
        current_hashes = {}
        for filepath in all_files:
            file_hash = self._get_file_hash(filepath)
            current_hashes[filepath] = file_hash
            if filepath not in indexed_files or indexed_files[filepath] != file_hash:
                new_or_modified.append(filepath)
        
        # If vectorstore exists and no new files, just load it
        if not new_or_modified and os.path.exists(self.persist_dir):
            print(f"[PhysRAG] All {len(all_files)} files already indexed. Loading existing ChromaDB...")
            vs = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
            try:
                self.total_chunks = vs._collection.count()
            except Exception:
                self.total_chunks = 0
            return vs
        
        # Load ALL documents (rebuild full index for consistency)
        documents = []
        for filepath in all_files:
            try:
                if filepath.endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")
                elif filepath.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"[PhysRAG Warning] Failed to load {filepath}: {e}")
            
        if not documents:
            print(f"[PhysRAG] No documents found in {self.data_dir}. System running with empty memory.")
            return Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
            
        print(f"[PhysRAG] Found {len(documents)} document pages from {len(all_files)} files.")
        print(f"[PhysRAG] New/modified files: {len(new_or_modified)}")
        
        # Optimized chunking: 800 chars with 200 overlap for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        self.total_chunks = len(chunks)
        
        print(f"[PhysRAG] Indexing {len(chunks)} chunks into ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        # Save index tracker
        self._save_indexed_files(current_hashes)
        print(f"[PhysRAG] Indexing complete! {len(chunks)} chunks from {len(all_files)} files.")
        
        return vectorstore

    def add_documents(self, file_paths: List[str]) -> int:
        """Add new documents to the knowledge base and reindex."""
        added = 0
        for path in file_paths:
            if os.path.exists(path):
                dest = os.path.join(self.data_dir, os.path.basename(path))
                if path != dest:
                    import shutil
                    shutil.copy2(path, dest)
                added += 1
        
        if added > 0:
            self.vectorstore = self._build_or_load_vectorstore()
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        return added

    def query_rag(self, user_query: str) -> Dict[str, Any]:
        """Performs RAG retrieval and synthesizes response using active LLM with performance tracking."""
        self.update_llm()
        
        # Timed retrieval
        start_time = time.time()
        retrieved_docs = self.retriever.invoke(user_query)
        self.last_retrieval_time = time.time() - start_time
        self.last_chunks_searched = len(retrieved_docs)
        
        sources = [doc.page_content for doc in retrieved_docs]
        source_metadata = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
        
        if not self.llm:
            fallback_answer = "[Notice] No active LLM API key detected. Showing direct ChromaDB vector search retrieval:\n\n### Vector Retrieval Results:\n\n" + "\n\n---\n\n".join(sources)
            return {
                "answer": fallback_answer,
                "retrieved_chunks": sources,
                "sources": list(set(source_metadata)),
                "retrieval_time": self.last_retrieval_time
            }
            
        system_template = """
        You are PhysRAG, an expert AI assistant specializing in:
        - Solid-State Physics & Condensed Matter Theory
        - Quantum Density Functional Theory (DFT) using Quantum Espresso, VASP, and Quantum ATK
        - ITO/Al2O3/Au Memristor Devices & Neuromorphic Computing Hardware
        - Space-borne Atomic Frequency Standards & Precision Metrology
        - Quantum Information Theory, Quantum Computing, and Quantum Error Correction
        
        Use the following retrieved context chunks from the PhysRAG knowledge base to answer the user's query accurately and rigorously.
        Always cite specific parameters, equations, and numerical values from the context when available.
        If the context does not fully answer the question, supplement with your physics knowledge but clearly indicate which parts come from the knowledge base vs. your general knowledge.
        
        RETRIEVED CONTEXT:
        {context}
        
        USER QUERY: {question}
        
        EXPERT ANSWER:
        """
        
        prompt = ChatPromptTemplate.from_template(system_template)
        
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)
            
        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            gen_start = time.time()
            answer = chain.invoke(user_query)
            gen_time = time.time() - gen_start
        except Exception as e:
            err_msg = str(e)
            answer = f"[Notice] API Invocation Notice: {err_msg}\n\n### Direct Vector Search Retrieval:\n\n" + "\n\n---\n\n".join(sources)
            gen_time = 0
                
        return {
            "answer": answer,
            "retrieved_chunks": sources,
            "sources": list(set(source_metadata)),
            "retrieval_time": self.last_retrieval_time,
            "generation_time": gen_time
        }

    def generate_dft_or_code_script(self, target_type: str, parameters: str) -> Dict[str, Any]:
        """RAG-augmented code generation: retrieves relevant KB context before generating code."""
        self.update_llm()
        if not self.llm:
            return {"code": "# Error: API Key missing or inactive.", "context_used": []}
        
        # First, retrieve relevant context from the knowledge base
        start_time = time.time()
        context_query = f"{target_type} parameters for {parameters}"
        retrieved_docs = self.retriever.invoke(context_query)
        retrieval_time = time.time() - start_time
        context_chunks = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
        You are a Computational Physics Code Generator with access to a specialized knowledge base.
        
        REFERENCE CONTEXT FROM KNOWLEDGE BASE:
        {context_text}
        
        TASK:
        Generate a complete, production-ready script for:
        Target Type: {target_type}
        Parameters/System: {parameters}
        
        INSTRUCTIONS:
        - If target_type contains 'Quantum Espresso', generate a valid .in SCF/NSCF input file with accurate parameters from the knowledge base context above.
        - If target_type contains 'Python Metrology', generate a Python script using NumPy, SciPy for Allan Variance / phase noise analysis.
        - If target_type contains 'Quantum Circuit', generate a Qiskit or Cirq quantum circuit script.
        - Use specific numerical values (cutoff energies, k-point grids, pseudopotentials) from the knowledge base when available.
        - Provide ONLY code with clear explanatory comments.
        """
        try:
            gen_start = time.time()
            response = self.llm.invoke(prompt)
            gen_time = time.time() - gen_start
            return {
                "code": response.content,
                "context_used": context_chunks,
                "retrieval_time": retrieval_time,
                "generation_time": gen_time
            }
        except Exception as e:
            return {"code": f"# API Error: {e}", "context_used": [], "retrieval_time": 0, "generation_time": 0}

    def get_kb_stats(self) -> Dict[str, Any]:
        """Returns knowledge base statistics."""
        files = self._discover_files()
        total_size = sum(os.path.getsize(f) for f in files)
        try:
            chunk_count = self.vectorstore._collection.count()
        except Exception:
            chunk_count = self.total_chunks
        
        return {
            "total_files": len(files),
            "total_size_kb": round(total_size / 1024, 1),
            "total_chunks": chunk_count,
            "file_details": [
                {
                    "name": os.path.basename(f),
                    "size_kb": round(os.path.getsize(f) / 1024, 1)
                }
                for f in files
            ]
        }

# Standalone test runner
if __name__ == "__main__":
    engine = PhysRAGEngine()
    print(f"\n[PhysRAG] Knowledge Base Stats: {engine.get_kb_stats()}")
    
    res = engine.query_rag("How does the Al2O3 barrier switch in memristors?")
    print("\n[PhysRAG Test Result]: Successfully initialized and queried ChromaDB vector database!")
    print("Active Provider:", engine.active_provider)
    print("Retrieved Chunks Count:", len(res["retrieved_chunks"]))
    print(f"Retrieval Time: {res['retrieval_time']:.3f}s")
    print("Answer Output Preview:\n", res["answer"][:300])
