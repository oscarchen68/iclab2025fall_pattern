#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_hlpte_vectors.py  (Generate merged golden data, formatted for PATTERN.v parsing)
Output:
  ./output/hlpte_patterns.json    (pattern definitions)
  ./output/hlpte_golden_all.txt   (merged golden, fixed format)
  ./output/input.txt              (original input format: FRAME + 16 INDEX lines)
  ./output/golden.txt             (same data as hlpte_golden_all but pure concatenated numeric values)
"""
import json, numpy as np
from pathlib import Path

# ----------------------------- user config -----------------------------
SEED = 2025
NUM_PATTERNS = 10
QP_RANGE = (0, 29)
FRAMES_SHAPE = (16, 32, 32)

out_dir = Path("./output")
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(SEED)

# -------------------------- quant tables --------------------------
MF_a_tbl = np.array([13107,11916,10082, 9362, 8192, 7282], dtype=np.int32)
MF_b_tbl = np.array([ 5243, 4660, 4194, 3647, 3355, 2893], dtype=np.int32)
MF_c_tbl = np.array([ 8066, 7490, 6554, 5825, 5243, 4559], dtype=np.int32)
V_a_tbl  = np.array([10,11,13,14,16,18], dtype=np.int32)
V_b_tbl  = np.array([16,18,20,23,25,29], dtype=np.int32)
V_c_tbl  = np.array([13,14,16,18,20,23], dtype=np.int32)

def qbits(qp): return 15 + (qp // 6)
def qp_div6(qp): return qp // 6
def qp_mod6(qp): return qp % 6
def quant_offset_f(qp):
    if qp <= 5:   return 10922
    if qp <= 11:  return 21845
    if qp <= 17:  return 43690
    if qp <= 23:  return 87381
    return 174762

def mf_selector(idx):
    if idx in (0,2,8,10): return 'a'
    if idx in (1,3,9,11,4,6,12,14): return 'c'
    if idx in (5,7,13,15): return 'b'
    return 'a'
def v_selector(idx): return mf_selector(idx)

# -------------------------- transform --------------------------
C = np.array([[1,1,1,1],
              [1,1,-1,-1],
              [1,-1,-1,1],
              [1,-1,1,-1]], dtype=np.int32)

def fwd4x4_int(X): return C @ X @ C.T
def inv4x4_int(Wp): return np.right_shift(C.T @ Wp @ C, 6).astype(np.int32)

def quant_4x4(W, qp):
    m6 = qp_mod6(qp)
    ma, mb, mc = MF_a_tbl[m6], MF_b_tbl[m6], MF_c_tbl[m6]
    f = quant_offset_f(qp); qb = qbits(qp)
    Z = np.empty((4,4), dtype=np.int32); flat=0
    for r in range(4):
        for c in range(4):
            M = {'a':ma,'b':mb,'c':mc}[mf_selector(flat)]
            w = int(W[r,c]); sgn = -1 if w < 0 else 1
            Z[r,c] = sgn * ((abs(w)*int(M) + f) >> qb)
            flat += 1
    return Z

def dequant_4x4(Z, qp):
    m6 = qp_mod6(qp)
    va, vb, vc = V_a_tbl[m6], V_b_tbl[m6], V_c_tbl[m6]
    sh = qp_div6(qp)
    Wp = np.empty((4,4), dtype=np.int32); flat=0
    for r in range(4):
        for c in range(4):
            V = {'a':va,'b':vb,'c':vc}[v_selector(flat)]
            Wp[r,c] = (int(Z[r,c]) * int(V)) << sh
            flat += 1
    return Wp

# -------------------------- predictors --------------------------
def pred_dc_tile(recon, gx0, gy0, is4x4, px, py, have_top, have_left):
    """DC prediction anchored to block/MB boundary (NOT per-pixel)."""
    if is4x4:
        sT = 0; sL = 0
        if have_top:
            top_y = gy0 - 1
            sT = int(recon[top_y, gx0:gx0+4].sum())
        if have_left:
            left_x = gx0 - 1
            sL = int(recon[gy0:gy0+4, left_x].sum())
        if have_top and have_left: return (sT + sL) >> 3  # /8
        elif have_top:             return sT >> 2         # /4
        elif have_left:            return sL >> 2         # /4
        else:                      return 128
    else:
        mbx = (gx0 // 16) * 16
        mby = (gy0 // 16) * 16
        sT = 0; sL = 0
        if have_top:
            top_y = mby - 1
            sT = int(recon[top_y, mbx:mbx+16].sum())
        if have_left:
            left_x = mbx - 1
            sL = int(recon[mby:mby+16, left_x].sum())
        if have_top and have_left: return (sT + sL) >> 5  # /32
        elif have_top:             return sT >> 4         # /16
        elif have_left:            return sL >> 4         # /16
        else:                      return 128


def pred_h_tile(recon, gx0, gy0, px, py, is4x4, have_left):
    if not have_left: return 128
    if is4x4:
        return int(recon[gy0 + py, gx0 - 1])
    else:
        mbx = (gx0 // 16) * 16
        return int(recon[gy0 + py, mbx - 1])

def pred_v_tile(recon, gx0, gy0, px, py, is4x4, have_top):
    if not have_top: return 128
    if is4x4:
        return int(recon[gy0 - 1, gx0 + px])
    else:
        mby = (gy0 // 16) * 16
        return int(recon[mby - 1, gx0 + px])


def pick_best_mode_4x4(src, recon, gx0, gy0):
    have_top  = (gy0 != 0)
    have_left = (gx0 != 0)

    # only valid candidates, with tie-break priority DC (0) < H (1) < V (2)
    candidates = [('DC', 0)]
    if have_left: candidates.append(('H', 1))
    if have_top:  candidates.append(('V', 2))

    sad = {name: 0 for name, _ in candidates}

    for py in range(4):
        for px in range(4):
            s = int(src[gy0+py, gx0+px])
            p_dc = pred_dc_tile(recon, gx0, gy0, True, px, py, have_top, have_left)
            sad['DC'] += abs(s - p_dc)
            if have_left:
                p_h = pred_h_tile(recon, gx0, gy0, px, py, True, have_left)
                sad['H'] += abs(s - p_h)
            if have_top:
                p_v = pred_v_tile(recon, gx0, gy0, px, py, True, have_top)
                sad['V'] += abs(s - p_v)

    items = [(name, sad[name], prio) for name, prio in candidates]
    items.sort(key=lambda x: (x[1], x[2]))
    return items[0][0], have_top, have_left

def pick_best_mode_16x16(src, recon, mbx, mby):
    have_top_mb  = (mby != 0)
    have_left_mb = (mbx != 0)

    candidates = [('DC', 0)]
    if have_left_mb: candidates.append(('H', 1))
    if have_top_mb:  candidates.append(('V', 2))

    sad = {name: 0 for name, _ in candidates}

    for py in range(16):
        for px in range(16):
            s = int(src[mby+py, mbx+px])
            p_dc = pred_dc_tile(recon, mbx, mby, False, px, py, have_top_mb, have_left_mb)
            sad['DC'] += abs(s - p_dc)
            if have_left_mb:
                p_h = pred_h_tile(recon, mbx, mby, px, py, False, have_left_mb)
                sad['H'] += abs(s - p_h)
            if have_top_mb:
                p_v = pred_v_tile(recon, mbx, mby, px, py, False, have_top_mb)
                sad['V'] += abs(s - p_v)

    items = [(name, sad[name], prio) for name, prio in candidates]
    items.sort(key=lambda x: (x[1], x[2]))
    return items[0][0], have_top_mb, have_left_mb

def build_pred_block(mode_name, recon, gx0, gy0, is4x4, have_top, have_left):
    P = np.empty((4,4), dtype=np.int32)
    for py in range(4):
        for px in range(4):
            if   mode_name == 'DC': v = pred_dc_tile(recon, gx0, gy0, is4x4, px, py, have_top, have_left)
            elif mode_name == 'H' : v = pred_h_tile (recon, gx0, gy0, px, py, is4x4, have_left)
            else                  : v = pred_v_tile (recon, gx0, gy0, px, py, is4x4, have_top)
            P[py, px] = v
    return P

# -------------------------- encode one INDEX (frame+modes+QP) ------------------
def encode_one(frames, index, modes4, qp, pattern_id=None, debug_target=(None,None), max_print=8):
    """
    frames: 16x32x32
    index: frame index to encode
    pattern_id: pattern number (1-based) passed from caller for debug
    debug_target: (pattern_id, frame_index) — when matches, prints debug
    max_print: how many 4x4 blocks to print
    """
    src   = frames[index].astype(np.uint8).copy()
    recon = np.zeros_like(src, dtype=np.uint8)
    Z_list = []
    printed = 0

    MB_coords = [(0,0), (16,0), (0,16), (16,16)]
    for mb_id, (mbx, mby) in enumerate(MB_coords):
        is4x4 = bool(modes4[mb_id])
        if not is4x4:
            mb_mode, have_top_mb, have_left_mb = pick_best_mode_16x16(src, recon, mbx, mby)
        else:
            mb_mode = None
            have_top_mb = have_left_mb = None

        for sby in range(4):
            for sbx in range(4):
                gx0 = mbx + sbx*4
                gy0 = mby + sby*4

                if is4x4:
                    mode_name, have_top4, have_left4 = pick_best_mode_4x4(src, recon, gx0, gy0)
                    ht, hl = have_top4, have_left4
                else:
                    mode_name = mb_mode
                    ht, hl = have_top_mb, have_left_mb

                P  = build_pred_block(mode_name, recon, gx0, gy0, is4x4=is4x4, have_top=ht, have_left=hl)
                X  = src[gy0:gy0+4, gx0:gx0+4].astype(np.int32) - P
                W  = fwd4x4_int(X)
                Z  = quant_4x4(W, qp)
                Z_list.extend(Z.reshape(-1).tolist())

                Wp = dequant_4x4(Z, qp)
                Xp = inv4x4_int(Wp).astype(np.int32)
                recon[gy0:gy0+4, gx0:gx0+4] = np.clip(Xp + P, 0, 255).astype(np.uint8)

                # debug printing:
                # if (pattern_id == debug_target[0]) and (index == debug_target[1]) and (printed < max_print):
                #     print(f"\n==== Pattern {pattern_id}  index {index}  Block #{printed}  pos=({gx0},{gy0}) ====")
                #     print(f"Mode: {mode_name} | is4x4={is4x4} | have_top={ht} have_left={hl}")
                #     # np.set_printoptions(threshold=np.inf, linewidth=200)
                #     print("P (Prediction block):\n", P)
                #     print("X (Residual):\n", X)
                #     print("W (Transformed):\n", W)
                #     print("Z (Quantized):\n", Z)
                #     print("Wp (dequant_4x4):\n", Wp)
                #     print("Xp (inv4x4_int):\n", Xp)
                #     # np.savetxt(f"src_p{pattern_id}_idx{index}.txt", src, fmt="%3d")
                #     # np.savetxt(f"recon_p{pattern_id}_idx{index}.txt", recon, fmt="%3d")
                #     printed += 1
                # else:
                #     printed += 1

    return np.array(Z_list, dtype=np.int32)


# -------------------------- generate patterns & per-pattern frames -----------------------
patterns = []
for s in range(NUM_PATTERNS):
    order = rng.permutation(16).tolist()
    params = []
    for i in range(16):
        idx    = order[i]
        modes4 = rng.randint(0, 2, size=(4,), dtype=np.int32).tolist()
        qp     = int(rng.randint(QP_RANGE[0], QP_RANGE[1]+1))
        params.append({"index": idx, "modes": modes4, "QP": qp})
    patterns.append({"params": params})

# with open(out_dir/"hlpte_patterns.json", "w", encoding="utf-8") as f:
#     json.dump(patterns, f, ensure_ascii=False, indent=2)

# We'll keep frames per-pattern so we can write input.txt correctly later
frames_per_pattern = []

# -------------------------- produce combined golden (single file) -----------
combined_path = out_dir / "hlpte_golden_all.txt"
with open(combined_path, "w", encoding="utf-8") as fout:
    fout.write("# HLPTE GOLDEN ALL v1\n")
    fout.write("NUM_PATTERNS %d\n\n" % len(patterns))
    simple_concat = []

    for p_idx, st in enumerate(patterns, start=1):
        # generate a fresh set of 16 frames for this pattern
        frames = rng.randint(0, 256, size=FRAMES_SHAPE).astype(np.uint8)
        frames_per_pattern.append(frames)  # save for input.txt later

        fout.write("PATTERN %d NUM_ENTRIES %d\n" % (p_idx, len(st["params"])))
        for e_idx, entry in enumerate(st["params"]):
            idx = int(entry["index"]); m0,m1,m2,m3 = entry["modes"]; qp = int(entry["QP"])
            # Generate Z and pass pattern_id, enable debug for (4,5)
            Z = encode_one(frames, idx, entry["modes"], qp,
                        pattern_id=p_idx,
                        debug_target=(4, 5),   # 4th pattern, index == 5
                        max_print=48)          # number of blocks to print

            fout.write(" ENTRY %d INDEX %d MODES %d %d %d %d QP %d LENGTH %d\n" %
                       (e_idx, idx, m0, m1, m2, m3, qp, len(Z)))
            for v in Z.tolist():
                fout.write("%d\n" % int(v))
            fout.write(" END_ENTRY\n")
            simple_concat.extend([int(v) for v in Z.tolist()])
        fout.write("END_PATTERN\n\n")

# write plain concatenated golden
with open(out_dir/"golden.txt", "w", encoding="utf-8") as fg:
    for v in simple_concat:
        fg.write("%d\n" % v)

# -------------------------- write input.txt (use frames_per_pattern) ------------------------------
with open(out_dir/"input.txt", "w", encoding="utf-8") as fi:
    fi.write("NUM_PATTERNS %d\n" % len(patterns))
    for p_idx, st in enumerate(patterns, start=1):
        fi.write("PATTERN %d\n" % p_idx)
        frames = frames_per_pattern[p_idx-1]
        for i in range(16):
            fi.write("FRAME %d\n" % i)
            tile = frames[i]
            for y in range(32):
                for x in range(32):
                    fi.write("%02X\n" % int(tile[y, x]))
        for entry in st["params"]:
            m0, m1, m2, m3 = entry["modes"]
            fi.write("INDEX %d MODES %d %d %d %d QP %d\n" % (entry["index"], m0, m1, m2, m3, entry["QP"]))
        fi.write("ENDPATTERN\n")

# -------------------------- summary prints -------------------------------
print("Wrote combined golden:", combined_path)
print("Wrote numeric golden:", out_dir/"golden.txt")
# print("Wrote patterns json :", out_dir/"hlpte_patterns.json")
print("Wrote input file    :", out_dir/"input.txt")
