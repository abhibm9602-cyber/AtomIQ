import os
import glob
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
    RAG & Agentic Engine for Solid-State Physics, DFT Simulations, and Precision Metrology.
    Integrates ChromaDB vector store with multi-provider LLM support:
    - Groq AI (Llama 3.3 70B - 100% Free Unrestricted API Tier)
    - Google Gemini
    - Anthropic Claude 3.5 Sonnet
    """
    def __init__(self, data_dir: str = "./data", persist_dir: str = "./chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # Initialize light local embedding model (no API key needed for vectorization!)
        print("[PhysRAG] Initializing HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load and Index Documents
        self.vectorstore = self._build_or_load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Setup LLM dynamically
        self.update_llm()

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
        """Loads text and PDF documents from data_dir and creates Chroma vector DB."""
        documents = []
        
        # Load .txt files
        for txt_file in glob.glob(os.path.join(self.data_dir, "*.txt")):
            loader = TextLoader(txt_file, encoding="utf-8")
            documents.extend(loader.load())
            
        # Load .pdf files
        for pdf_file in glob.glob(os.path.join(self.data_dir, "*.pdf")):
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            
        if not documents:
            print(f"[PhysRAG] No documents found in {self.data_dir}. System running with empty memory.")
            return Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
            
        print(f"[PhysRAG] Found {len(documents)} document pages. Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        print(f"[PhysRAG] Indexing {len(chunks)} chunks into ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        return vectorstore

    def query_rag(self, user_query: str) -> Dict[str, Any]:
        """Performs RAG retrieval and synthesizes response using active LLM with error handling."""
        self.update_llm()
        retrieved_docs = self.retriever.invoke(user_query)
        sources = [doc.page_content for doc in retrieved_docs]
        source_metadata = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
        
        if not self.llm:
            fallback_answer = "[Notice] No active LLM API key detected. Showing direct ChromaDB vector search retrieval:\n\n### Vector Retrieval Results:\n\n" + "\n\n---\n\n".join(sources)
            return {
                "answer": fallback_answer,
                "retrieved_chunks": sources,
                "sources": list(set(source_metadata))
            }
            
        system_template = """
        You are PhysRAG, an expert AI assistant specializing in Solid-State Physics, Quantum Density Functional Theory (DFT), ITO/Al2O3/Au Memristor Devices, and Space-borne Atomic Frequency Standards.
        
        Use the following retrieved context chunks to answer the user's scientific query accurately and rigorously.
        If the context does not fully answer the question, use your physics knowledge to provide a comprehensive answer, but cite the retrieved context where applicable.
        
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
            answer = chain.invoke(user_query)
        except Exception as e:
            err_msg = str(e)
            answer = f"[Notice] API Invocation Notice: {err_msg}\n\n### Direct Vector Search Retrieval:\n\n" + "\n\n---\n\n".join(sources)
                
        return {
            "answer": answer,
            "retrieved_chunks": sources,
            "sources": list(set(source_metadata))
        }

    def generate_dft_or_code_script(self, target_type: str, parameters: str) -> str:
        """Agentic tool to auto-generate Quantum Espresso input scripts or Python Allan variance analysis code."""
        self.update_llm()
        if not self.llm:
            return "# Error: API Key missing or inactive."
            
        prompt = f"""
        You are a Computational Physics Code Generator.
        Generate a complete, production-ready script for:
        Target Type: {target_type}
        Parameters/System: {parameters}
        
        If target_type is 'Quantum Espresso', generate a valid `.in` SCF/NSCF input file.
        If target_type is 'Python Metrology', generate a Python script using NumPy, SciPy, and Allan Variance calculation for atomic clock signal stability.
        
        Provide ONLY code with clear explanatory comments.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"# API Error: {e}"

# Standalone test runner
if __name__ == "__main__":
    engine = PhysRAGEngine()
    res = engine.query_rag("How does the Al2O3 barrier switch in memristors?")
    print("\n[PhysRAG Test Result]: Successfully initialized and queried ChromaDB vector database!")
    print("Active Provider:", engine.active_provider)
    print("Retrieved Chunks Count:", len(res["retrieved_chunks"]))
    print("Answer Output Preview:\n", res["answer"][:300])
