import sys
if sys.platform.startswith('linux'):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

import os
import glob
import time
import hashlib
import json
import re
from typing import Dict, List, Any, Optional
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

class AtomIQEngine:
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
        self.mp_api_key: Optional[str] = os.getenv("MP_API_KEY")
        
        # Initialize light local embedding model (no API key needed for vectorization!)
        print("[AtomIQ] Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
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
        
        # Priority 1: Groq AI (OpenAI GPT-OSS 120B - Free & Powerful)
        if groq_key and groq_key.strip():
            try:
                self.llm = ChatGroq(
                    model_name="openai/gpt-oss-120b",
                    groq_api_key=groq_key.strip(),
                    temperature=0.2
                )
                self.active_provider = "Groq GPT-OSS 120B (Free)"
                return
            except Exception as e:
                print(f"[AtomIQ Warning] Failed with Groq: {e}")

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
                    print(f"[AtomIQ Warning] Failed with Gemini {model_name}: {e}")
                    
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
                print(f"[AtomIQ Warning] Failed to initialize Anthropic LLM: {e}")

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
            print(f"[AtomIQ] All {len(all_files)} files already indexed. Loading existing ChromaDB...")
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
                print(f"[AtomIQ Warning] Failed to load {filepath}: {e}")
            
        if not documents:
            print(f"[AtomIQ] No documents found in {self.data_dir}. System running with empty memory.")
            return Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
            
        print(f"[AtomIQ] Found {len(documents)} document pages from {len(all_files)} files.")
        print(f"[AtomIQ] New/modified files: {len(new_or_modified)}")
        
        # Optimized chunking: 800 chars with 200 overlap for better context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        self.total_chunks = len(chunks)
        
        print(f"[AtomIQ] Indexing {len(chunks)} chunks into ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        
        # Save index tracker
        self._save_indexed_files(current_hashes)
        print(f"[AtomIQ] Indexing complete! {len(chunks)} chunks from {len(all_files)} files.")
        
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
        You are AtomIQ, an expert AI assistant specializing in:
        - Solid-State Physics & Condensed Matter Theory
        - Quantum Density Functional Theory (DFT) using Quantum Espresso, VASP, and Quantum ATK
        - ITO/Al2O3/Au Memristor Devices & Neuromorphic Computing Hardware
        - Space-borne Atomic Frequency Standards & Precision Metrology
        - Quantum Information Theory, Quantum Computing, and Quantum Error Correction
        
        Use the following retrieved context chunks from the AtomIQ knowledge base to answer the user's query accurately and rigorously.
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
            # Strip Qwen's <think> reasoning tags from the answer
            if '</think>' in answer:
                answer = answer.split('</think>', 1)[1].strip()
            elif '<think>' in answer:
                answer = answer.replace('<think>', '').strip()
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

    def fetch_materials_data(self, query: str) -> Optional[str]:
        """Query the Materials Project API for real crystallographic data."""
        if not self.mp_api_key:
            return None
        
        # Extract chemical formula from the query using common patterns
        formula_match = re.search(r'\b([A-Z][a-z]?(?:\d*[A-Z][a-z]?)*\d*)\b', query)
        if not formula_match:
            return None
        
        formula = formula_match.group(1)
        # Basic validation: must have at least one uppercase letter
        if not any(c.isupper() for c in formula) or len(formula) < 2:
            return None
        
        try:
            from mp_api.client import MPRester
            with MPRester(self.mp_api_key) as mpr:
                # Search for the material by formula
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=[
                        "material_id", "formula_pretty", "structure",
                        "symmetry", "band_gap", "formation_energy_per_atom",
                        "energy_above_hull", "is_stable"
                    ]
                )
                
                if not docs:
                    return None
                
                # Use the most stable entry (lowest energy above hull)
                doc = sorted(docs, key=lambda x: x.energy_above_hull if x.energy_above_hull is not None else 999)[0]
                
                lattice = doc.structure.lattice
                
                mp_text = f"""LIVE MATERIALS PROJECT DATA (mp-api) for {doc.formula_pretty}:
- Materials Project ID: {doc.material_id}
- Formula: {doc.formula_pretty}
- Crystal System: {doc.symmetry.crystal_system if doc.symmetry else 'N/A'}
- Space Group: {doc.symmetry.symbol if doc.symmetry else 'N/A'}
- Space Group Number: {doc.symmetry.number if doc.symmetry else 'N/A'}
- Lattice Parameters:
  a = {lattice.a:.4f} Angstrom
  b = {lattice.b:.4f} Angstrom
  c = {lattice.c:.4f} Angstrom
  alpha = {lattice.alpha:.2f} degrees
  beta  = {lattice.beta:.2f} degrees
  gamma = {lattice.gamma:.2f} degrees
- Volume: {lattice.volume:.4f} Angstrom^3
- Band Gap: {doc.band_gap:.3f} eV
- Formation Energy: {doc.formation_energy_per_atom:.4f} eV/atom
- Energy Above Hull: {doc.energy_above_hull:.4f} eV/atom
- Thermodynamically Stable: {doc.is_stable}
- Number of Sites: {len(doc.structure)}
- Atomic Positions (fractional):"""
                for site in doc.structure:
                    mp_text += f"\n  {site.species_string}: ({site.frac_coords[0]:.6f}, {site.frac_coords[1]:.6f}, {site.frac_coords[2]:.6f})"
                
                return mp_text
                
        except ImportError:
            print("[AtomIQ] mp-api not installed. Skipping Materials Project lookup.")
            return None
        except Exception as e:
            print(f"[AtomIQ] Materials Project API error: {e}")
            return None

    def generate_dft_or_code_script(self, target_type: str, parameters: str) -> Dict[str, Any]:
        """RAG-augmented dual-agent code generation: Generator + Critic + Live Materials Project data."""
        self.update_llm()
        if not self.llm:
            return {"code": "# Error: API Key missing or inactive.", "initial_code": "", "critique": "", "context_used": [], "mp_data": None}
        
        # First, retrieve relevant context from the knowledge base
        start_time = time.time()
        context_query = f"{target_type} parameters for {parameters}"
        retrieved_docs = self.retriever.invoke(context_query)
        retrieval_time = time.time() - start_time
        context_chunks = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Fetch live Materials Project data if API key is available
        mp_data = self.fetch_materials_data(parameters)
        if mp_data:
            context_text = f"{mp_data}\n\n===== LOCAL KNOWLEDGE BASE CONTEXT =====\n\n{context_text}"
        
        # AGENT 1: GENERATOR
        prompt_gen = f"""/no_think
You are a Computational Physics Code Generator with access to a specialized knowledge base.

REFERENCE CONTEXT FROM KNOWLEDGE BASE:
{context_text}

TASK:
Generate a complete, production-ready script for:
Target Type: {target_type}
Parameters/System: {parameters}

CRITICAL RULES:
- Do NOT include any thinking, reasoning, or explanation. Output ONLY the script code.
- Do NOT use <think> tags.
- If target_type contains 'Quantum Espresso', generate a valid .in SCF/NSCF input file with accurate parameters.
- If target_type contains 'Python', generate a valid Python script.
- Use specific numerical values from the knowledge base when available.
- Provide ONLY code with clear explanatory comments. Do not wrap the output in markdown code blocks.
- Start your response directly with the code. No preamble.
"""
        
        try:
            gen_start = time.time()
            initial_response = self.llm.invoke(prompt_gen)
            initial_code = initial_response.content
            
            # AGENT 2: PHYSICS CRITIC
            prompt_critic = f"""/no_think
You are a Senior Computational Physics Reviewer. Review the script below and fix any physical inconsistencies, parameter mismatches, or syntactic errors.

TARGET SCRIPT TYPE: {target_type}
SYSTEM PARAMETERS: {parameters}

INITIAL DRAFT CODE:
{initial_code}

REFERENCE KNOWLEDGE BASE CONTEXT:
{context_text}

CRITICAL RULES:
- Do NOT include any thinking or reasoning process. Do NOT use <think> tags.
- You MUST follow the EXACT format below. No other format is acceptable.
- Start your response IMMEDIATELY with the word CRITIQUE:

FORMAT (follow exactly):
CRITIQUE:
[2-4 sentences about physical flaws or confirming correctness]

REFINED CODE:
[The complete corrected code here, no markdown blocks]
"""
            
            critic_response = self.llm.invoke(prompt_critic)
            critic_text = critic_response.content
            gen_time = time.time() - gen_start
            
            # Strip Qwen's <think> reasoning tags from both responses
            def _strip_think_tags(text):
                # If there's a closing </think> tag, keep only what comes AFTER it
                if '</think>' in text:
                    text = text.split('</think>', 1)[1].strip()
                # If there's only an opening <think> with no close, remove just the tag
                elif '<think>' in text:
                    text = text.replace('<think>', '').strip()
                return text
            
            initial_code = _strip_think_tags(initial_code)
            critic_text = _strip_think_tags(critic_text)
            
            # Parse the critic's response
            if "REFINED CODE:" in critic_text:
                parts = critic_text.split("REFINED CODE:")
                critique = parts[0].replace("CRITIQUE:", "").strip()
                refined_code = parts[1].strip()
            else:
                critique = "Critic validated the initial draft."
                refined_code = initial_code
                
            # Clean up markdown code blocks if the critic hallucinated them
            if refined_code.startswith("```"):
                lines = refined_code.split("\n")
                if len(lines) > 2:
                    refined_code = "\n".join(lines[1:-1])

            return {
                "initial_code": initial_code,
                "critique": critique,
                "code": refined_code,
                "context_used": context_chunks,
                "retrieval_time": retrieval_time,
                "generation_time": gen_time,
                "mp_data": mp_data
            }
        except Exception as e:
            return {"code": f"# API Error: {e}", "initial_code": "", "critique": "", "context_used": [], "retrieval_time": 0, "generation_time": 0, "mp_data": None}

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
    engine = AtomIQEngine()
    print(f"\n[AtomIQ] Knowledge Base Stats: {engine.get_kb_stats()}")
    
    res = engine.query_rag("How does the Al2O3 barrier switch in memristors?")
    print("\n[AtomIQ Test Result]: Successfully initialized and queried ChromaDB vector database!")
    print("Active Provider:", engine.active_provider)
    print("Retrieved Chunks Count:", len(res["retrieved_chunks"]))
    print(f"Retrieval Time: {res['retrieval_time']:.3f}s")
    print("Answer Output Preview:\n", res["answer"][:300])
