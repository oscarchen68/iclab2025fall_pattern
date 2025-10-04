# ICLAB 2025 Fall — Lab04 CNN Testbench

This folder (`00_TESTBED`) contains the **complete testing environment** for **Lab04: CNN (Convolutional Neural Network)** in ICLAB 2025 Fall.  
It includes the input/golden vector generator, automatic testbench (`PATTERN.v`), and simulation integration with your `CNN.v` DUT.



## Directory Structure

00_TESTBED/
├── PATTERN.v # Main pattern generator / checker (Verilog testbench)
├── TESTBED.v # Top-level testbench wrapper (instantiates CNN + PATTERN)
├── gen_cnn_vectors.py # Python script to generate input.txt / golden.txt
├── README.md # This documentation file


## Overview

Lab04 has **two tasks**:
- **Task0:** CNN classification pipeline (Conv + Pool + Activation + FC + Softmax)
- **Task1:** Knapsack-like optimization using convolutional feature scores

## File Descriptions

### `PATTERN.v`
- Provides full **automatic verification** for both Task0 and Task1.
- Checks **SPEC compliance**:
  - SPEC-4 → Reset behavior (`out_valid`, `out` must be 0)
  - SPEC-5 → when out_valid==0, outputs must be zero
  - SPEC-6 → Latency ≤ 150 cycles
  - SPEC-8 → Random gap (2–4 negedges) between cases
- Handles **input feeding**, **timing synchronization**, and **golden result comparison**.
- Supports IEEE-754 **float32 comparison** (error tolerance ≤ 1e-6).

### `gen_cnn_vectors.py`
- Python script that **generates random test vectors** and corresponding golden outputs.
- Supports both tasks with correct data layout for PATTERN and DUT.

#### Key options:
```bash
python gen_cnn_vectors.py \
    --cases 100 \
    --task0-ratio 0.5 \
    --seed 42 \
    --input input.txt \
    --golden golden.txt
```

- `--cases` : number of total testcases.
- `--task0-ratio` : probability of Task0 (0.0–1.0). The rest are Task1.
- `--seed` : random seed (for reproducibility).
- Generates:
  - `input.txt` — the input vectors consumed by PATTERN.v
  - `golden.txt` — the expected golden results


TESTBED.v
- Top-level wrapper that instantiates:
  - `CNN.v` (your design)
  - `PATTERN.v` (the verifier)
- Handles waveform generation and simulation setup.


## Example Command Summary

# Step 1. Generate random input/golden vectors
```
python3 gen_cnn_vectors.py --cases 8 --task0-ratio 0.5 --seed 123
```

# Step 2. Run your Verilog simulation
```
vcs -full64 -sverilog ../CNN.v TESTBED.v PATTERN.v -o simv
./simv | tee log.txt
```
or
```
./01_run_vcs_rtl
```

# Step 3. Check the result
grep "PASS" log.txt
```
```