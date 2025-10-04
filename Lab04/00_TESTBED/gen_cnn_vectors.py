#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for ICLAB CNN Lab (Lab04) vectors:
- Writes input.txt and output.txt (golden) in the format used by your PATTERN.
- Supports Task0 and Task1, computes true golden values.

Usage:
  python3 gen_cnn_vectors.py --cases 1000 --seed 123 --task0-ratio 0.5 --input ./input.txt --golden ./golden.txt

  python3 gen_cnn_vectors.py --cases 1000 --seed 123 --task0-ratio 0 --input ./input.txt --golden ./golden.txt

Notes:
- mode bits: mode[1]=padding (0=replication,1=reflection), mode[0]=activation (0=tanh,1=swish).
"""

import argparse
import struct
import numpy as np
import random
from pathlib import Path

# -------------------------
# Float helpers (IEEE-754)
# -------------------------
def f32(x):  # ensure numpy float32
    return np.float32(x)

def f32_to_hex(val_f32: np.float32) -> str:
    """Convert float32 to uppercase 8-hex without '0x'."""
    b = struct.pack('>f', float(val_f32))
    u = struct.unpack('>I', b)[0]
    return f"{u:08X}"

# -------------------------
# Core ops
# -------------------------
def pad_img(img6, pad_mode):
    """Return 8x8 padded image. pad_mode: 0 replication('edge'), 1 reflection('reflect')."""
    mode = 'edge' if pad_mode == 0 else 'reflect'
    return np.pad(img6, 1, mode=mode).astype(np.float32)

def conv2d_single(padded8, k3):
    """6x6 valid conv on 8x8 padded input with 3x3 kernel."""
    out = np.zeros((6,6), dtype=np.float32)
    for y in range(6):
        for x in range(6):
            win = padded8[y:y+3, x:x+3]
            out[y,x] = f32(np.sum(win * k3))
    return out

def pool3x3_stride3_max(feat6):
    """3x3 max-pool with stride=3 -> 2x2."""
    pooled = np.zeros((2,2), dtype=np.float32)
    for py, y0 in enumerate([0,3]):
        for px, x0 in enumerate([0,3]):
            pooled[py,px] = np.max(feat6[y0:y0+3, x0:x0+3]).astype(np.float32)
    return pooled

def activate(x, act_mode):
    """Activation: 0=tanh, 1=swish."""
    x = x.astype(np.float32)
    if act_mode == 0:
        return np.tanh(x).astype(np.float32)
    else:
        return (x / (1.0 + np.exp(-x))).astype(np.float32)
        

def leaky_relu(x, alpha=0.01):
    return np.where(x>=0, x, alpha*x).astype(np.float32)

def softmax3(z3):
    z = z3.astype(np.float32)
    m = np.max(z).astype(np.float32)
    e = np.exp((z - m).astype(np.float32)).astype(np.float32)
    s = np.sum(e).astype(np.float32)
    return (e / s).astype(np.float32)

# -------------------------
# Random data
# -------------------------
def random_image(channels, rng):
    if channels == 1:
        return f32(rng.uniform(-0.5, 0.5, (6,6)))
    else:
        return (f32(rng.uniform(-0.5, 0.5, (6,6))), f32(rng.uniform(-0.5, 0.5, (6,6))))

def random_kernel(rng):
    return f32(rng.uniform(-0.5, 0.5, (3,3)))

def random_fc_weights(shape, rng):
    return f32(rng.normal(-0.5, 0.5, shape))

def random_fc_bias_scalar(rng):
    """Return a single scalar bias (float32)."""
    return f32(rng.normal(-0.5, 0.5))

def flatten_pooled_2x2(p):
    # row-major: [0,0], [0,1], [1,0], [1,1]
    return f32(p.reshape(-1))

def write_line(fh, s):
    fh.write((s + "\n").encode())

# -------------------------
# Task0 generator + golden
# -------------------------
def task0_compute_and_emit(f_in, f_gold, mode_bits, rng, wb_count=63):
    """
    Task0:
      - Image: 6x6x2
      - Kernels: two 3x3x2 (k0_ch1/ch2, k1_ch1/ch2)
      - FC: 5x8 -> LeakyReLU(0.01) -> 3x5 -> softmax
    Emits:
      header, 72 image words (interleaved ch0,ch1), 18 kernel lines (ch1 ch2),
      WB (57 lines), then golden 3 lines.
    """
    pad_mode = (mode_bits >> 1) & 1
    act_mode = (mode_bits >> 0) & 1

    img0, img1 = random_image(2, rng)
    k0_ch1 = random_kernel(rng); k0_ch2 = random_kernel(rng)
    k1_ch1 = random_kernel(rng); k1_ch2 = random_kernel(rng)

    # FC (full layout)
    FC1_W = random_fc_weights((5,8), rng)
    FC1_B = random_fc_bias_scalar(rng)   # single scalar bias for FC1
    # print("[DEBUG] FC1_W:\n", FC1_W)
    # print("[DEBUG] FC1_B:\n", FC1_B)

    FC2_W = random_fc_weights((3,5), rng)
    FC2_B = random_fc_bias_scalar(rng)   # single scalar bias for FC2
    # print("[DEBUG] FC2_W:\n", FC2_W)
    # print("[DEBUG] FC2_B:\n", FC2_B)

    # ---- forward ----
    P0 = pad_img(img0, pad_mode); P1 = pad_img(img1, pad_mode)
    # print("[DEBUG] img0:\n", img0)
    # print("[DEBUG] img1:\n", img1)
    # print("[DEBUG] P0 (padded img0):\n", P0)
    # print("[DEBUG] P1 (padded img1):\n", P1)

    conv0 = conv2d_single(P0, k0_ch1) + conv2d_single(P1, k0_ch2)
    conv1 = conv2d_single(P0, k1_ch1) + conv2d_single(P1, k1_ch2)
    # print("[DEBUG] conv0 result:\n", conv0)
    # print("[DEBUG] conv1 result:\n", conv1)

    pool0 = pool3x3_stride3_max(conv0)
    pool1 = pool3x3_stride3_max(conv1)
    # print("[DEBUG] pool0 result:\n", pool0)
    # print("[DEBUG] pool1 result:\n", pool1)

    act0  = activate(pool0, act_mode)
    act1  = activate(pool1, act_mode)
    # print("[DEBUG] act0 result:\n", act0)
    # print("[DEBUG] act1 result:\n", act1)

    feat  = np.concatenate([flatten_pooled_2x2(act0), flatten_pooled_2x2(act1)], axis=0)  # (8,)
    # print("[DEBUG] feat vector (8,):\n", feat)

    fc1   = (FC1_W @ feat + FC1_B).astype(np.float32)   # FC1_B broadcasts (scalar)
    # print("[DEBUG] fc1 result:\n", fc1)

    lrelu = leaky_relu(fc1, 0.01)
    # print("[DEBUG] lrelu result:\n", lrelu)

    dot_vals = (FC2_W @ lrelu).astype(np.float32)
    # print("[DEBUG] FC2_W @ lrelu:\n", dot_vals)

    logits = (dot_vals + FC2_B).astype(np.float32)
    # print("[DEBUG] logits:\n", logits)

    prob   = softmax3(logits)

    # ---- emit input ----
    write_line(f_in, f"0 {mode_bits}")

    # Image: first entire channel 0, then entire channel 1
    for y in range(6):
        for x in range(6):
            write_line(f_in, f32_to_hex(img0[y,x]))

    for y in range(6):
        for x in range(6):
            write_line(f_in, f32_to_hex(img1[y,x]))
    
    # Kernels: 18 lines (first 9 → kernel0(ch1 ch2), next 9 → kernel1(ch1 ch2))
    for i in range(3):
        for j in range(3):
            write_line(f_in, f"{f32_to_hex(k0_ch1[i,j])} {f32_to_hex(k0_ch2[i,j])}")
    for i in range(3):
        for j in range(3):
            write_line(f_in, f"{f32_to_hex(k1_ch1[i,j])} {f32_to_hex(k1_ch2[i,j])}")

    # ---- emit WB (57 layout: FC1_W + FC1_B + FC2_W + FC2_B) ----
    for r in range(5):
        for c in range(8):
            write_line(f_in, f32_to_hex(FC1_W[r,c]))
    # write single scalar FC1_B
    write_line(f_in, f32_to_hex(FC1_B))
    for r in range(3):
        for c in range(5):
            write_line(f_in, f32_to_hex(FC2_W[r,c]))
    # write single scalar FC2_B
    write_line(f_in, f32_to_hex(FC2_B))

    # ---- emit golden (3 lines) ----
    for k in range(3):
        write_line(f_gold, f32_to_hex(prob[k]))

# -------------------------
# Task1 generator + golden
# -------------------------
def task1_compute_and_emit(f_in, f_gold, mode_bits, rng):
    """
    Task1:
      - Image: 6x6x1
      - Kernels: A..D (four 3x3)
      - conv → 3x3 maxpool/stride=3 → activation → sum (per kernel)
      - 0/1 knapsack (capacity, costA..D) → pick best mask (tie-break: higher value; then lower cost; then smaller mask)
    Emits:
      header, 36 image words, 18 kernel lines (A/B then C/D), 5 decimal lines (cap,costA..D),
      and golden 1 line (32-bit hex with low-4 bits=mask).
    """
    pad_mode = (mode_bits >> 1) & 1
    act_mode = (mode_bits >> 0) & 1

    img = random_image(1, rng)
    kA = random_kernel(rng); kB = random_kernel(rng)
    kC = random_kernel(rng); kD = random_kernel(rng)

    cap   = random.randint(1, 15)
    # print(f"cap = {cap}")
    costA = random.randint(1, 15)
    costB = random.randint(1, 15)
    costC = random.randint(1, 15)
    costD = random.randint(1, 15)
    costs  = [costD, costC, costB, costA]
    # print(f"costs = {costs}")
    

    P = pad_img(img, pad_mode)

    def kernel_score(k3, name):
        conv = conv2d_single(P, k3)
        # print(f"[DEBUG] Kernel {name} (3x3):\n", k3)
        # print(f"[DEBUG] conv {name} (6x6):\n", conv)
        score = f32(np.sum(conv))
        # print(f"[DEBUG] score {name}: {score}\n")
        return score

    scores = [
        kernel_score(kD, "D"),
        kernel_score(kC, "C"),
        kernel_score(kB, "B"),
        kernel_score(kA, "A")
    ]
    # print("[DEBUG] all scores:", scores)

    total_score = sum(scores)
    if all(s < 0 for s in scores):
        # print("[DEBUG] all scores < 0, force best_mask=0000")
        best_mask  = 0
        best_value = f32(0.0)
        best_cost  = 0
    else:
        # knapsack over 4 items
        best_mask  = 0
        best_value = f32(-1e30)
        best_cost  = 10**9
        for mask in range(16):
            w = 0
            v = f32(0.0)
            for i in range(4):
                if mask & (1<<i):
                    w += costs[i]
                    v = f32(v + scores[i])
            if w <= cap:
                if (v > best_value + 1e-6) or (abs(float(v - best_value)) <= 1e-6 and (w < best_cost or (w == best_cost and mask < best_mask))):
                    best_value = f32(v)
                    best_cost  = w
                    best_mask  = mask

    # ---- emit input ----
    write_line(f_in, f"1 {mode_bits}")
    for y in range(6):
        for x in range(6):
            write_line(f_in, f32_to_hex(img[y,x]))
    # First 9 lines A/B, next 9 lines C/D
    for i in range(3):
        for j in range(3):
            write_line(f_in, f"{f32_to_hex(kA[i,j])} {f32_to_hex(kC[i,j])}")
    for i in range(3):
        for j in range(3):
            write_line(f_in, f"{f32_to_hex(kB[i,j])} {f32_to_hex(kD[i,j])}")
    for t in [cap, costA, costB, costC, costD]:
        write_line(f_in, str(int(t)))

    # ---- emit golden ----
    gold32 = best_mask & 0xF
    # print(f"gold32 = {gold32}")
    write_line(f_gold, f"{gold32:08X}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6, help="number of testcases")
    ap.add_argument("--task0-ratio", type=float, default=0.5, help="probability of selecting Task0 for a case")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--wb-count", type=int, default=57, choices=[57,63], help="Task0 Weight_Bias lines (63=spec, 57=compact)")
    ap.add_argument("--input", type=str, default="input.txt", help="output path for input.txt")
    ap.add_argument("--golden", type=str, default="output.txt", help="output path for output.txt")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)

    inp = Path(args.input); gol = Path(args.golden)
    with inp.open("wb") as f_in, gol.open("wb") as f_gold:
        for _ in range(args.cases):
            is_t0 = (random.random() < args.task0_ratio)
            mode_bits = random.randint(0, 3)
            if is_t0:
                task0_compute_and_emit(f_in, f_gold, mode_bits, rng, wb_count=args.wb_count)
            else:
                task1_compute_and_emit(f_in, f_gold, mode_bits, rng)

    print(f"[OK] Generated:\n  {inp}\n  {gol}")

if __name__ == "__main__":
    main()

"""
===============================================================================
INPUT / GOLDEN FORMAT SPECIFICATION  (for 00_TESTBED/PATTERN.v)
-------------------------------------------------------------------------------
Overview:
  - PATTERN.v reads testcases from INPUT_FILE ("../00_TESTBED/input.txt")
    and compares DUT outputs against GOLDEN_FILE ("../00_TESTBED/golden.txt").
  - Each testcase begins with a 1-line header: "<task> <mode>"
      * task : 0 = Task0, 1 = Task1
      * mode : integer 0..3 (2-bit)
          - mode[1] = padding   -> 0: replication, 1: reflection
          - mode[0] = activation-> 0: tanh,        1: swish
  - After the header, the file contains a fixed sequence of lines whose
    length and meaning depend on the task field. All numeric tokens are
    expected in hexadecimal for 32-bit data (see notes below). For capacity/cost
    lines use decimal integers 0..15.

-------------------------------------------------------------------------------
TASK0 (header line: "0 <mode>")
  - Purpose: 6x6x2 input image, two 3x3x2 kernels, FC weights & biases
  - Line sequence after header:
      1) Image: 72 lines, each a 32-bit IEEE-754 single (hex). Order:
           image uses raster order, channel-interleaved:
           - We expect the testbench to feed 72 cycles in this sequence.
           - Convention used by PATTERN.v: Image words are provided in the
             same order the DUT expects (raster-scan). Example mapping:
               row0 col0 ch0
               row0 col0 ch1
               row0 col1 ch0
               row0 col1 ch1
               ...
               row5 col5 ch1
      2) Kernel: 18 lines, each line contains TWO hex tokens:
           "<Kernel_ch1> <Kernel_ch2>"
           - First 9 lines → Kernel0 (3x3) ch1+ch2 (kernel0_ch1[0..8], kernel0_ch2[0..8])
           - Next 9 lines  → Kernel1 (3x3) ch1+ch2 (kernel1_ch1[0..8], kernel1_ch2[0..8])
           - Each line is two 32-bit hex values separated by whitespace.
      3) Weight_Bias: 57 lines, each a single 32-bit hex token.
           The expected layout (order must match PATTERN.v and DUT):
             - FC1_W [5 rows] × [8 cols]  : (5*8 = 40 words)
                 Order in file: FC1_W[0][0..7], FC1_W[1][0..7], ..., FC1_W[4][0..7]
             - FC1_B [1]                  : FC1_B  (1 words)
             - FC2_W [3 rows] × [5 cols]  : (3*5 = 15 words)
                 Order in file: FC2_W[0][0..4], FC2_W[1][0..4], FC2_W[2][0..4]
             - FC2_B [1]                  : FC2_B[0], FC2_B[1], FC2_B[2]  (1 words)
           (Total = 40 + 1 + 15 + 1 = 57)
  - Golden (in golden.txt):
      - For each Task0 case, provide exactly 3 lines, each a single 32-bit hex
        representing the expected softmax outputs (IEEE-754 single).
      - Order: softmax_class0, softmax_class1, softmax_class2

-------------------------------------------------------------------------------
TASK1 (header line: "1 <mode>")
  - Purpose: 6x6x1 input image, four 3x3 kernels (A..D), knapsack capacity/cost
  - Line sequence after header:
      1) Image: 36 lines, each a 32-bit hex (raster order; single channel)
      2) Kernel: 18 lines, each line contains TWO hex tokens:
           "<Kernel_ch1> <Kernel_ch2>"
           - First 9 lines → kernels A and C (packed: ch1->A, ch2->C)
           - Next 9 lines  → kernels B and D (packed: ch1->B, ch2->D)
      3) capacity_cost: 5 lines, each a decimal integer (0..15):
           - Line 0: capacity
           - Line 1: costA
           - Line 2: costB
           - Line 3: costC
           - Line 4: costD
  - Golden (in golden.txt):
      - For each Task1 case, provide exactly 1 line: a 32-bit hex word.
      - Formatting: high 28 bits must be 0; low 4 bits encode chosen mask.
        Example: if chosen mask = 0b0101 => golden = 0x00000005

-------------------------------------------------------------------------------
NUMERIC FORMATTING AND EXAMPLES
  - 32-bit hex tokens: either "3f800000" or "0x3f800000" are acceptable for humans;
    PATTERN.v uses "%h" fscanf which accepts typical hex token formats. To be
    safe, prefer no "0x" prefix or be consistent across the entire file.
  - Decimal tokens (for capacity/cost): write plain integers, e.g. "7" or "0".
  - Example minimal Task0 block (illustrative, not real weights):

    // input.txt (excerpt)
    0 1
    3f800000
    00000000
    ... (total 72 image lines) ...
    3f000000 3f800000   // kernel line 0 (ch1 ch2)
    3f000001 3f800001
    ... (total 18 kernel lines) ...
    3e800000            // weight/bias line 0
    3e700000
    ... (total 57 weight/bias lines) ...

    // golden.txt (corresponding golden excerpt)
    3f19999a
    3f000000
    3ecccccd

  - Example minimal Task1 block:

    // input.txt (excerpt)
    1 0
    3f800000
    00000000
    ... (36 image lines) ...
    3f000000 3f800000  // kernel line 0
    ...
    10                 // capacity (decimal)
    2
    3
    1
    4

    // golden.txt (golden)
    00000005           // chosen mask in low-4 bits (example)

-------------------------------------------------------------------------------
"""