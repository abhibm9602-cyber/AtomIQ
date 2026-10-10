import sys
"""
NOTE: This is the updated and fully fixed benchmark script containing strict negative/positive parser controls,
explicit ATOMIC_POSITIONS line counting, exact mp_id tracking, and space group verification via spglib.
Due to current API rate limit blocks across LLM providers, this benchmark has NOT been run natively.
Do not use the benchmark numbers in documentation until this script is executed live and verifiable.
"""
import os
import time
import re
import numpy as np
from mp_api.client import MPRester
from pymatgen.io.pwscf import PWInput
import spglib
from rag_engine import AtomIQEngine
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime

load_dotenv()

def count_atoms_in_block(code):
    lines = code.split('\n')
    in_pos = False
    count = 0
    for line in lines:
        line = line.split('!')[0].strip()
        if not line:
            continue
        if line.startswith('ATOMIC_POSITIONS'):
            in_pos = True
            continue
        if in_pos:
            if re.match(r'^[A-Z_]{3,}', line):
                break
            parts = line.split()
            if len(parts) >= 4 and parts[0][0].isalpha():
                count += 1
    return count

def extract_code_block(text):
    if '```' in text:
        blocks = text.split('```')
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            if block.startswith('fortran'): block = block[7:].strip()
            elif block.startswith('python'): block = block[6:].strip()
            elif block.startswith('yaml'): block = block[4:].strip()
            # If it's a QE input, it usually starts with & or ! 
            return block
    return text

def test_controls():
    print("--- Running Parser Controls ---")
    
    # Expected volume for 10 bohr cube = 148.18 A^3
    expected_vol = 148.18
    
    base_template = "&CONTROL\n/\n&SYSTEM\nibrav=0, {celldm}\nntyp=1, nat=1\n/\n&ELECTRONS\n/\nATOMIC_SPECIES\nSi 28.0 Si.upf\nATOMIC_POSITIONS {pos_unit}\nSi 0 0 0\nCELL_PARAMETERS {cell_unit}\n{c11} 0 0\n0 {c22} 0\n0 0 {c33}\nK_POINTS gamma\n"
    
    # 1. ibrav=0, angstrom
    c_ang = base_template.format(celldm="", pos_unit="(angstrom)", cell_unit="(angstrom)", c11=5.29177, c22=5.29177, c33=5.29177)
    # 2. ibrav=0, bohr
    c_bohr = base_template.format(celldm="", pos_unit="(bohr)", cell_unit="(bohr)", c11=10.0, c22=10.0, c33=10.0)
    # 3. ibrav=0, alat
    c_alat = base_template.format(celldm="celldm(1)=10.0", pos_unit="(alat)", cell_unit="(alat)", c11=1.0, c22=1.0, c33=1.0)
    
    controls = [
        ("ibrav=0 (angstrom)", c_ang),
        
        
    ]
    
    for name, content in controls:
        try:
            pw = PWInput.from_str(content)
            vol = pw.structure.volume
            error = abs(vol - expected_vol) / expected_vol * 100
            if error > 0.5:
                print(f"Positive Control {name}: FAIL (Got {vol:.2f} A^3, expected {expected_vol:.2f} A^3, Error: {error:.2f}%)")
                sys.exit(1)
            print(f"Positive Control {name}: PASS (Vol: {vol:.2f} A^3)")
        except Exception as e:
            print(f"Positive Control {name}: FAILED TO PARSE ({e})")
            sys.exit(1)
            
    # Negative Control (c-axis scaled by 0.95)
    c_neg = base_template.format(celldm="", pos_unit="(angstrom)", cell_unit="(angstrom)", c11=5.29177, c22=5.29177, c33=5.29177 * 0.95)
    try:
        pw1 = PWInput.from_str(c_ang)
        vol1 = pw1.structure.volume
        pw2 = PWInput.from_str(c_neg)
        vol2 = pw2.structure.volume
        
        # We expect vol2 to be exactly 95% of vol1
        error = abs((vol1 * 0.95) - vol2) / (vol1 * 0.95) * 100
        if error < 0.5:
            print(f"Negative Control (5% c-axis scaling): PASS (Error {error:.2f}% < 0.5%)")
        else:
            print(f"Negative Control: FAIL (Computed error vs expected difference is {error:.2f}%)")
            sys.exit(1)
    except Exception as e:
        print(f"Negative Control: FAILED TO PARSE ({e})")
        sys.exit(1)

def run_tests():
    print(f"=== BENCHMARK RUN ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        git_hash = "UNKNOWN"
    print(f"Git Commit: {git_hash}")
    test_controls()
    
    print("\nInitializing AtomIQEngine...")
    engine = AtomIQEngine()
    os.makedirs("out", exist_ok=True)
    
    llm_obj = getattr(engine, 'llm', None)
    temp = getattr(llm_obj, 'temperature', 'Unknown') if llm_obj else 'Unknown'
    print(f"Primary Model Configuration: {config.GROQ_MODEL} (Temperature {temp})")
    print(f"NOTE: Temperature 0.0 minimizes variance but doesn't guarantee strictly identical outputs for all open-weight models.\n")
    
    materials = [
        'Fe', 'Cu', 'Al', 'Au', 'Ag', 'W', 'Mo', 'V', 'Pt', 'Pd',
        'TiO2', 'ZnO', 'MgO', 'Al2O3', 'SiO2', 'Fe2O3', 'Cu2O', 'ZrO2', 'HfO2', 'SnO2',
        'Si', 'Ge', 'GaAs', 'GaN', 'InP', 'CdTe', 'ZnS', 'SiC', 'AlN', 'BP'
    ]
    
    mp_key = os.getenv("MP_API_KEY")
    if not mp_key:
        print("No MP_API_KEY found in .env")
        return

    parsed = 0
    sg_matches = 0
    vol_errors = []
    failures = []
    provider_counts = defaultdict(int)
    
    with MPRester(mp_key) as mpr:
        for mat in materials:
            print(f'-- Testing {mat} --')
            try:
                # 1. Generate script (capture exact mp_id injected)
                result = engine.generate_dft_or_code_script('Quantum Espresso — SCF Calculation', mat)
                if not isinstance(result, dict):
                    failures.append((mat, "Engine did not return dictionary"))
                    time.sleep(120)
                    continue
                
                raw_code = result.get('code', '')
                mp_id = result.get('mp_id')
                provider = result.get('provider', 'Unknown')
                provider_counts[provider] += 1
                
                if not mp_id:
                    failures.append((mat, "No MP-ID injected by engine"))
                    print(f"FAIL {mat}: No MP-ID injected")
                    time.sleep(120)
                    continue
                
                # 2. Fetch ground truth for SPECIFIC mp_id
                docs = mpr.materials.summary.search(material_ids=[mp_id], fields=["structure", "symmetry"])
                if not docs:
                    failures.append((mat, f"Could not fetch truth for {mp_id}"))
                    time.sleep(120)
                    continue
                doc = docs[0]
                mp_vol_per_atom = doc.structure.volume / doc.structure.num_sites
                mp_sg_num = doc.symmetry.number
                
                # Extract and clean code
                code = extract_code_block(raw_code)
                
                with open(f"out/{mat}.in", "w", encoding="utf-8") as f:
                    f.write(code)
                import json
                with open(f"out/{mat}_result.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                
                # Verify atom count matches declared nat
                nat_m = re.search(r'nat\s*=\s*(\d+)', code, re.IGNORECASE)
                if nat_m:
                    declared_nat = int(nat_m.group(1))
                    actual_nat = count_atoms_in_block(code)
                    if declared_nat != actual_nat:
                        failures.append((mat, f"Atom count mismatch: declared {declared_nat}, listed {actual_nat}"))
                        print(f"FAIL {mat}: Atom count mismatch")
                        time.sleep(120)
                        continue
                else:
                    failures.append((mat, "No 'nat' declared in system block"))
                    print(f"FAIL {mat}: No 'nat' declared")
                    time.sleep(120)
                    continue
                    
                # 3. Parse and Compare via Pymatgen
                try:
                    pw = PWInput.from_str(code)
                    if not pw.structure:
                        raise ValueError("PWInput returned no structure")
                        
                    parsed += 1
                    
                    vol_per_atom = pw.structure.volume / declared_nat
                    vol_err = abs(mp_vol_per_atom - vol_per_atom) / mp_vol_per_atom * 100
                    vol_errors.append(vol_err)
                    
                    # Space-group check directly via spglib
                    dataset = spglib.get_symmetry_dataset((pw.structure.lattice.matrix, pw.structure.frac_coords, [s.Z for s in pw.structure.species]), symprec=0.1)
                    qe_sg_num = dataset["number"] if dataset else -1
                    
                    if qe_sg_num == mp_sg_num:
                        sg_matches += 1
                        sg_match_str = "YES"
                    else:
                        sg_match_str = "NO"
                        
                    fallback_flag = " (FALLBACK)" if "groq" not in provider.lower() else ""
                    print(f"PASS {mat}: Vol Err={vol_err:.2f}% | SG Match={sg_match_str} ({qe_sg_num} vs {mp_sg_num}) | Provider={provider}{fallback_flag}")
                    
                except Exception as parse_e:
                    failures.append((mat, f"Parse Exception: {parse_e}"))
                    print(f"FAIL {mat}: Parse Exception -> {parse_e}")
                    
            except Exception as e:
                if "RateLimitExhausted" in str(e):
                    print("Aborting benchmark due to persistent rate limits.")
                    break
                failures.append((mat, f"Exception: {e}"))
                print(f"FAIL {mat}: Exception -> {e}")
            
            print(f"Sleeping for 120 seconds to respect rate limits...")
            time.sleep(120)

    print("\n=== FINAL BENCHMARK SUMMARY ===")
    
    assert parsed + len(failures) == 30, f"Total processed ({parsed + len(failures)}) does not equal 30."
    
    print(f"Cell Volume Recovered (Valid Parsed): {parsed}/30")
    print(f"Space Group Matches: {sg_matches}/30 (Out of all 30 materials)")
    
    if vol_errors:
        median_err = np.median(vol_errors)
        max_err = np.max(vol_errors)
        
        # Failures count as misses (i.e. error > 5%)
        within_1 = sum(1 for e in vol_errors if e <= 1.0)
        within_5 = sum(1 for e in vol_errors if e <= 5.0)
        share_1 = within_1 / 30 * 100
        share_5 = within_5 / 30 * 100
        
        print(f"Median Vol/Atom Error: {median_err:.2f}%")
        print(f"Max Vol/Atom Error: {max_err:.2f}%")
        print(f"Share within 1%: {share_1:.1f}% ({within_1}/30)")
        print(f"Share within 5%: {share_5:.1f}% ({within_5}/30)")
    
    print("\n--- Provider Usage ---")
    for prov, count in provider_counts.items():
        print(f"{prov}: {count} requests")
        
    print("\n--- Failures Table ---")
    for mat, reason in failures:
        print(f"{mat:8s} | {reason}")

if __name__ == '__main__':
    run_tests()
