from rag_engine import AtomIQEngine
e = AtomIQEngine()
print("Formula:", e._extract_formula("MACE — Training Script (mace_run_train) Si"))
print("AFLOW:", e.fetch_aflow_data("Si"))
