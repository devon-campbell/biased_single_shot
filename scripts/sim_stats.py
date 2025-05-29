from tqdm import tqdm
import json
import time
import numpy as np
from ldpc.codes import ring_code
from ldpc.code_util import compute_code_distance
import ldpc.mod2 as mod2
from ldpc import protograph as pt
from ldpc.codes import hamming_code
from bposd.css import css_code
from bposd.hgp import hgp
from css_ss_decode_sim import css_ss_decode_sim
from lifted_hgp_4d import *
from bposd.css_decode_sim import css_decode_sim
import logging
from functools import reduce
from z3 import Optimize, Bool, Xor, Sum, If, Or, is_true, sat, Solver
import datetime
from distance import compute_min_hamming_distance
from mm_qc_pega.graphs import QC_tanner_graph
from ldpc import BpOsdDecoder
import scipy
from collections import Counter
import itertools
import os
import joblib
import numpy as np
import argparse
from ldpc import protograph as pt
from mm_qc_pega.graphs import QC_tanner_graph
from lifted_hgp_4d import lifted_hgp_4d
import os
from sweep_grid_v1 import grid
from joblib import Parallel, delayed
import time, random, os
import pandas as pd

RESULTS_CSV = "../results/results_v4.csv"
TARGET_RUNS = 750
N_JOBS = -1  # all cores
codename = 'peg_6_4_3_0-1'
qcode = joblib.load(f"../codes/{codename}/qcode.pkl.z")

def get_sim_stats(sim):
    S = sim.stats
    out = {}

    for basis in ("x", "z"):
        TP = S[f"TP{basis}"]; FN = S[f"FN{basis}"]
        FP = S[f"FP{basis}"]; TN = S[f"TN{basis}"]
        TC = S[f"TC{basis}"]; MC = S[f"MC{basis}"]

        meas_err_count = TP + FN
        det_rate   = TP / (TP + FN) if TP + FN else 0.0
        false_pos  = FP / (FP + TN) if FP + TN else 0.0
        corr_rate  = TC / (TP + FN) if TP + FN else 0.0
        miscorr    = MC / (TP + FP) if TP + FP else 0.0

        out.update({
            f'meas_err_count_{basis}': meas_err_count,
            f'det_rate_{basis}': det_rate,
            f'false_pos_{basis}': false_pos,
            f'corr_rate_{basis}': corr_rate,
            f'miscorr_{basis}': miscorr
        })

    return out

def simulate_one(params):
    err, meas, bias, hadamard, ss, apply_ss, chan = params
    seed = random.randint(0, 2**32 - 1)

    sim_input = {
        "error_rate": err,
        "p_meas_err": meas,
        "xyz_error_bias": bias,
        "hadamard_rotate": hadamard,
        "hadamard_rotate_sector1_length": qcode.hx1.shape[1],
        "run_ss": ss,
        "apply_ss":apply_ss,
        "channel_update": chan,
        "target_runs": TARGET_RUNS,
        "bp_method": "minimum_sum",
        "ms_scaling_factor": 0.625,
        "osd_method": "osd_cs",
        "osd_order": 4, 
        "max_iter": int(qcode.N / 10),
        "tqdm_disable": True,
    }

    try:
        start = time.time()
        sim = css_ss_decode_sim(hx=qcode.hx, hz=qcode.hz, mx=qcode.mx, mz=qcode.mz, **sim_input)
        end = time.time()

        input_dict = {
            "error_rate": err,
            "p_meas_err": meas,
            "xyz_bias": str(bias),
            "hadamard_rotate": hadamard,
            "run_ss": ss,
            "apply_ss": apply_ss,
            "channel_update": chan,
            "seed": seed,
            "sec_per_run": (end - start) / TARGET_RUNS,
        }

        metrics = {
            'd_max':sim.min_logical_weight,
            'OSDW_WER': sim.osd0_word_error_rate,
            'OSDW': sim.osdw_logical_error_rate,
            'OSD0': sim.osd0_logical_error_rate,
        }
        
        ss_stats = get_sim_stats(sim) if ss else {}
        return input_dict | metrics | ss_stats

    except Exception as e:
        return {
            "error_rate": err,
            "p_meas_err": meas,
            "xyz_bias": str(bias),
            "hadamard_rotate": hadamard,
            "run_ss": ss,
            "apply_ss": apply_ss,
            "channel_update": chan,
            "seed": seed,
            "error": str(e)
        }

def save_results(results):
    df = pd.DataFrame(results)
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

def main():
    CHUNK = 100  # Number of sims per batch (adjustable)
    random.shuffle(grid)
    for i in range(0, len(grid), CHUNK):
        batch = grid[i:i+CHUNK]
        print(f"Running batch {i}–{i+len(batch)-1}", flush=True)
        results = Parallel(n_jobs=N_JOBS)(delayed(simulate_one)(params) for params in tqdm(batch))
        save_results(results)

if __name__ == "__main__":
    main()