# âš›ï¸ AtomIQ

> **Autonomous Multi-Agent AI Framework for Computational Materials Science, DFT Simulation Automation, and Physics-Driven Code Generation**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Groq Llama 3.3](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat&logo=meta&logoColor=white)](https://groq.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![Anthropic Claude](https://img.shields.io/badge/Anthropic-Claude%203.5-D97706?style=flat&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F61?style=flat)](https://www.trychroma.com/)
[![Materials Project](https://img.shields.io/badge/API-Materials%20Project-2CA02C?style=flat)](https://materialsproject.org/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

Developed by **Abhijith Krishnan B M** â€” B.Tech-M.S. in Solid State Physics, Indian Institute of Space Science and Technology (IIST) | Former ISRO Space Applications Centre.

---

## ðŸ“¸ Overview

**AtomIQ** is an autonomous AI framework that bridges solid-state physics domain expertise with modern generative AI architectures. It employs a **dual-agent system** (Code Generator + Physics Critic) augmented by a **Retrieval-Augmented Generation (RAG)** pipeline to autonomously generate, validate, and refine production-ready simulation scripts across 9 major domains of computational physics.

Unlike generic code generation tools, AtomIQ retrieves domain-specific knowledge from an indexed physics knowledge base and pulls **live crystallographic data** from the Materials Project, AFLOW, OQMD, and PubChem APIs â€” ensuring every generated script uses physically accurate parameters (lattice constants, space groups, atomic positions, band gaps) rather than placeholder values.

### What Makes AtomIQ Different

| Feature | Generic AI Coding Tools | AtomIQ |
|---|---|---|
| **Physics grounding** | None â€” hallucinated parameters | RAG-augmented with indexed physics literature |
| **Materials data** | Static, often incorrect | **Live Materials Project, AFLOW, OQMD, and PubChem APIs** â€” real lattice constants, band gaps |
| **Validation** | No physics validation | **Dual-agent system** â€” Physics Critic reviews every script |
| **Domain coverage** | General-purpose | 9 specialized physics domains, 53 script types |
| **LLM flexibility** | Single provider | **3 providers** â€” Groq (free), Gemini, Claude |

---

## ðŸ—ï¸ Architecture

```mermaid
flowchart TD
    A["ðŸ“„ Physics Knowledge Base<br/>(15 documents, 160 vector chunks)"] -->|HuggingFace Embeddings<br/>all-MiniLM-L6-v2| B["ðŸ§  ChromaDB Vector Store"]
    C["ðŸ” User Query / Material System"] --> D{"AtomIQ RAG Engine"}
    B -->|Semantic Retrieval<br/>Top-5 chunks| D
    MP["ðŸŒ Materials Project, AFLOW, OQMD, and PubChem APIs<br/>(Live crystallographic data)"] -->|mp-api query| D
    D -->|Context-enriched prompt| E["ðŸ¤– Agent 1: Code Generator"]
    E -->|Initial draft| F["ðŸ•µï¸ Agent 2: Physics Critic"]
    F -->|Refined, validated script| G["âœ¨ Production-Ready Output"]
    D -->|Direct RAG response| H["ðŸ“– Literature Synthesis"]
    
    style MP fill:#2CA02C,color:#fff
    style E fill:#4F8BF9,color:#fff
    style F fill:#F59E0B,color:#fff
    style G fill:#10B981,color:#fff
```

### The Dual-Agent Workflow

1. **RAG Retrieval** â€” The knowledge base is searched for relevant physics context (DFT parameters, material properties, simulation methodologies)
2. **Materials Project Query** â€” If a chemical formula is detected, AtomIQ fetches live crystallographic data (lattice parameters, space group, band gap, formation energy, atomic positions)
3. **Agent 1 (Code Generator)** â€” Generates a complete simulation script using the retrieved context and live materials data
4. **Agent 2 (Physics Critic)** â€” Reviews the script for physical inconsistencies, parameter mismatches, and computational errors, then produces the final refined code

---

## âš¡ Code Generation â€” 8 Domains, 53 script types

### ðŸ”¬ 1. DFT & Electronic Structure (7 scripts)

First-principles calculations using Quantum Espresso and VASP â€” the workhorses of computational materials science.

| Script | Description |
|---|---|
| **Quantum Espresso â€” SCF Calculation** | Self-consistent field calculation with accurate pseudopotentials, k-point grids, and energy cutoffs |
| **Quantum Espresso â€” Band Structure** | Electronic band structure along high-symmetry paths in the Brillouin zone |
| **Quantum Espresso â€” DOS / PDOS** | Total and projected density of states for electronic structure analysis |
| **Quantum Espresso â€” Phonon Calculation (DFPT)** | Density Functional Perturbation Theory for phonon dispersion and vibrational properties |
| **Quantum Espresso â€” Structural Relaxation** | Geometry optimization using BFGS algorithm with variable cell relaxation |
| **VASP â€” INCAR/POSCAR/KPOINTS Generation** | Complete VASP input file set with appropriate ENCUT, EDIFF, ISMEAR, and k-mesh settings |
| **Python â€” DFT Post-Processing (pymatgen)** | Automated post-processing of DFT output files using the pymatgen library |

### ðŸ§ª 2. Molecular Dynamics (4 scripts)

Classical and ab initio molecular dynamics simulations for thermal, mechanical, and transport properties.

| Script | Description |
|---|---|
| **LAMMPS â€” Metal Simulation (EAM)** | Embedded Atom Method simulations for metallic systems â€” stress-strain, defect dynamics |
| **LAMMPS â€” Thermal Conductivity (Green-Kubo)** | Non-equilibrium MD for computing lattice thermal conductivity via autocorrelation functions |
| **Python â€” MD with ASE** | Molecular dynamics using the Atomic Simulation Environment â€” NVT/NPT ensembles |
| **Python â€” Radial Distribution Function** | Pair correlation function $g(r)$ analysis for structural characterisation |

### ðŸ“Š 3. Data Analysis & Precision Metrology (6 scripts)

Signal processing, spectroscopy analysis, and precision measurement tools â€” informed by ISRO atomic clock metrology experience.

| Script | Description |
|---|---|
| **Python â€” Allan Variance / Allan Deviation** | Time-domain frequency stability analysis $\sigma_y(\tau)$ for atomic frequency standards |
| **Python â€” Phase Noise Analysis** | Power spectral density analysis of oscillator phase fluctuations |
| **Python â€” XRD Pattern Simulation** | Powder X-ray diffraction pattern generation from crystal structure data |
| **Python â€” Tauc Plot Band Gap Extraction** | Optical band gap determination from UV-Vis spectroscopy data using Tauc relation |
| **Python â€” IV Curve Analysis (Memristor)** | Current-voltage hysteresis loop analysis for resistive switching devices |
| **Python â€” Impedance Spectroscopy (Nyquist Plot)** | AC impedance analysis and equivalent circuit fitting for electrochemical systems |

### ðŸ’» 4. Quantum Computing (5 scripts)

Quantum circuit construction and simulation using IBM Qiskit and Google Cirq frameworks.

| Script | Description |
|---|---|
| **Qiskit â€” Quantum Teleportation Circuit** | Complete quantum teleportation protocol with Bell pair preparation and classical communication |
| **Qiskit â€” Grover's Search Algorithm** | Oracle construction and amplitude amplification for unstructured database search |
| **Qiskit â€” VQE (Variational Quantum Eigensolver)** | Hybrid quantum-classical ground state energy estimation with parameterized ansatz |
| **Qiskit â€” Bell State Preparation & Measurement** | EPR pair creation and Bell basis measurement for entanglement verification |
| **Cirq â€” Quantum Circuit Simulation** | Google Cirq-based circuit construction and state vector simulation |

### ðŸ§  5. Machine Learning for Molecules (5 scripts)

AI/ML pipelines for molecular property prediction and high-throughput screening.

| Script | Description |
|---|---|
| **Python â€” Molecular Graph Neural Network (MPNN)** | Graph neural network for predicting molecular properties from chemical structure |
| **Python â€” HOMO-LUMO Gap Prediction (Random Forest)** | Supervised ML model for electronic gap prediction from molecular features |
| **Python â€” Molecular Binding Energy Prediction** | Regression model for thermodynamic stability assessment |
| **Python â€” PubChem API Data Fetch** | Automated bulk data retrieval from the PubChem database via API |
| **Python â€” SMILES to RDKit Descriptor Calculation** | Molecular environment descriptor generation for ML potentials |

### ðŸ“ 6. Statistical Mechanics (5 scripts)

Computational statistical mechanics and thermodynamic simulations.

| Script | Description |
|---|---|
| **Python â€” 2D Ising Model (Monte Carlo)** | Metropolis Monte Carlo simulation of magnetic phase transitions on a square lattice |
| **Python â€” Metropolis Algorithm Simulation** | General-purpose Metropolis-Hastings sampling for thermodynamic ensembles |
| **Python â€” Fermi-Dirac Distribution Plotter** | Temperature-dependent Fermi-Dirac occupation function visualisation |
| **Python â€” Phonon Density of States** | Vibrational density of states from dynamical matrix diagonalisation |
| **Python â€” Boltzmann Transport Equation Solver** | Semiclassical electronic transport calculation â€” electrical and thermal conductivity |

---

## ðŸŒ Live Materials Project, AFLOW, OQMD, and PubChem APIs Integration

When a chemical formula is detected in the user's input, AtomIQ automatically queries the Materials Project database and injects **real crystallographic data** into the code generation pipeline:

```yaml
LIVE MATERIALS PROJECT DATA for Al2O3:
  Materials Project ID: mp-1143
  Formula: Al2O3
  Crystal System: trigonal
  Space Group: R-3c
  Lattice Parameters:
    a: 4.8054 Ã…, b: 4.8054 Ã…, c: 13.1176 Ã…
    Î±: 90.00Â°, Î²: 90.00Â°, Î³: 120.00Â°
  Band Gap: 5.850 eV
  Formation Energy: -3.4420 eV/atom
  Thermodynamically Stable: True
  Atomic Positions: [fractional coordinates for all sites]
```

This ensures that every Quantum Espresso or VASP input file uses **experimentally validated** lattice constants and atomic positions â€” not LLM-hallucinated values.

---

## ðŸ“š Knowledge Base (15 Documents, 160 Vector Chunks)

The RAG knowledge base spans the following domains:

| Domain | Document | Topics Covered |
|---|---|---|
| **Condensed Matter** | `condensed_matter_theory.txt` | Band theory, Fermi surfaces, BCS superconductivity, Cooper pairs |
| **DFT Methods** | `dft_computational_methods.txt` | Exchange-correlation functionals, pseudopotentials, convergence testing |
| **Memristors** | `memristor_switching_mechanisms.txt` | Resistive switching, filamentary conduction, oxide barriers |
| **Memristor + Metrology** | `memristor_and_metrology_ref.txt` | ITO/Alâ‚‚Oâ‚ƒ/Au heterostructures, tunneling transport |
| **ITO/TCO Materials** | `ito_tco_materials.txt` | Transparent conducting oxides, thin film deposition |
| **Neuromorphic Hardware** | `neuromorphic_computing_hardware.txt` | Crossbar arrays, synaptic devices, in-memory computing |
| **Atomic Clocks** | `atomic_clocks_metrology_advanced.txt` | Rubidium frequency standards, Allan deviation, clock stability |
| **Semiconductor Devices** | `semiconductor_device_physics.txt` | Fowler-Nordheim tunneling, MOS physics, band offsets |
| **Quantum Information** | `quantum_information_theory.txt` | Qubits, entanglement, Bell inequalities, quantum error correction |
| **Quantum Field Theory** | `quantum_field_theory_fundamentals.txt` | Path integrals, symmetry breaking, Higgs mechanism |
| **QFT Textbook** | Peskin & Schroeder (PDF) | Complete QFT reference â€” Feynman diagrams, renormalisation |
| **Statistical Mechanics** | `statistical_mechanics_thermodynamics.txt` | Partition functions, Ising model, phase transitions |
| **ML for Materials** | `machine_learning_materials_science.txt` | CGCNN, MACE, NequIP, DeePMD, ML interatomic potentials |
| **Molecular Dynamics** | `molecular_dynamics_simulations.txt` | Velocity-Verlet, thermostats, barostats, Green-Kubo |
| **Spectroscopy** | `signal_processing_spectroscopy.txt` | Tauc plots, XRD analysis, impedance spectroscopy |

---

## ðŸ”§ Multi-Provider LLM Support

AtomIQ automatically detects and connects to the best available LLM provider:

| Provider | Model | Cost | Speed |
|---|---|---|---|
| **Groq** | Llama 3.3 70B / GPT-OSS 120B | **Free** (no API charges) | âš¡ Ultra-fast (Groq LPU) |
| **Google** | Gemini 2.0 Flash / Pro | Free tier available | Fast |
| **Anthropic** | Claude 3.5 Sonnet | Paid | Highest quality |

The system falls through providers automatically â€” if Groq is unavailable, it tries Gemini, then Claude. No configuration change needed.

---

## ðŸ› ï¸ Quick Start

### 1. Clone & Set Up
```bash
git clone https://github.com/abhibm9602-cyber/AtomIQ.git
cd AtomIQ
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_groq_key_here          # Free at console.groq.com
MP_API_KEY=your_materials_project_key_here    # Free at materialsproject.org
# Optional:
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_claude_key
```

### 3. Run
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ðŸ“„ Repository Structure
```
AtomIQ/
â”œâ”€â”€ app.py                  # Streamlit web interface with 4 tabs
â”œâ”€â”€ rag_engine.py           # Core engine â€” RAG, dual-agent, Materials Project, AFLOW, OQMD, and PubChem APIs
â”œâ”€â”€ requirements.txt        # Python dependencies
â”œâ”€â”€ Run_AtomIQ_App.bat      # Windows one-click launcher
â”œâ”€â”€ data/                   # Physics knowledge base (15 documents)
â”‚   â”œâ”€â”€ condensed_matter_theory.txt
â”‚   â”œâ”€â”€ dft_computational_methods.txt
â”‚   â”œâ”€â”€ memristor_switching_mechanisms.txt
â”‚   â”œâ”€â”€ atomic_clocks_metrology_advanced.txt
â”‚   â”œâ”€â”€ quantum_information_theory.txt
â”‚   â”œâ”€â”€ machine_learning_materials_science.txt
â”‚   â”œâ”€â”€ molecular_dynamics_simulations.txt
â”‚   â”œâ”€â”€ statistical_mechanics_thermodynamics.txt
â”‚   â””â”€â”€ ... (15 files total)
â”œâ”€â”€ chroma_db/              # Persistent ChromaDB vector store
â””â”€â”€ README.md
```

---

## ðŸ”¬ Technical Specifications

| Component | Detail |
|---|---|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | ChromaDB with persistent disk storage |
| **Chunk Size** | 800 characters, 200-character overlap |
| **Retrieval** | Top-5 semantic similarity search |
| **Materials API** | Materials Project v2 (`mp-api`) |
| **Frontend** | Streamlit with custom CSS theming |
| **Agent Architecture** | Dual-agent (Generator â†’ Critic) with RAG context injection |

---

## ðŸ¤ Author

**Abhijith Krishnan B M**  
B.Tech-M.S. in Solid State Physics | Indian Institute of Space Science and Technology (IIST)  
Former Intern, ISRO Space Applications Centre â€” Rubidium Atomic Clock Calibration & Precision Metrology

- [GitHub](https://github.com/abhibm9602-cyber) | [Email](mailto:abhi.bm9602@gmail.com)
