import os
import sys

from rag_engine import AtomIQEngine

engine = AtomIQEngine()
print("Engine loaded.")
result = engine.generate_dft_or_code_script("MACE — Training Script (mace_run_train)", "Si")
print("\n--- INITIAL CODE ---\n")
print(result.get("initial_code", ""))
print("\n--- CRITIQUE ---\n")
print(result.get("critique", ""))
print("\n--- REFINED CODE ---\n")
print(result.get("code", ""))
print("\n--- DB DATA ---\n")
print(result.get("mp_data", ""))
