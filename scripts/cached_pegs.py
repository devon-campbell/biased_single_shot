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
from ldpc import BpOsdDecoder
import scipy
from collections import Counter
import itertools
import os
from mm_qc_pega.graphs import QC_tanner_graph
import yaml
import pickle

# Load QC protograph from PEG generator
def qc_proto_to_pt(proto_matrix, N):
    """
    Convert QC_tanner_graph.proto to ldpc.protograph.array,
    """
    m, n = proto_matrix.shape
    obj = np.empty((m, n), dtype=object)

    for i in range(m):
        for j in range(n):
            s = int(proto_matrix[i, j])
            obj[i, j] = () if s == -1 else ((-s) % N,)

    return pt.array(obj)

with open("../mm_qc_pega/config.yml", "r") as ymlfile:
    settings = yaml.safe_load(ymlfile)
    path = settings["output_fname"]
    codename = path.split("/")[-1].split(".")[0]

G = QC_tanner_graph.read(f"../mm_qc_pega/{path}")
A_p = qc_proto_to_pt(G.proto, N=G.N)         # convert to pt.array

# sanity check: binary matrices must match
H_qc   = G.get_H()
H_pt   = A_p.to_binary(lift_parameter=G.N)
assert np.array_equal(H_qc, H_pt)

A, B, C, D = get_blocks(A_p, A_p, A_p, A_p, codename)

mz_proto = maybe_cached("mz", codename, lambda: construct_MZ(A, B, C, D))
hz1_proto = maybe_cached("hz1", codename, lambda: construct_HZ(A, B, C, D)[0])
hz2_proto = maybe_cached("hz2", codename, lambda: construct_HZ(A, B, C, D)[1])
hx1_proto = maybe_cached("hx1", codename, lambda: construct_HX(A, B, C, D)[0])
hx2_proto = maybe_cached("hx2", codename, lambda: construct_HX(A, B, C, D)[1])
mx_proto  = maybe_cached("mx", codename, lambda: construct_MX(A, B, C, D))

mz_proto, hz1_proto, hz2_proto, hx1_proto, hx2_proto, mx_proto = construct_4d_matrices(A_p, A_p, A_p, A_p, codename)

qcode = lifted_hgp_4d(G.N, a=A_p, b=A_p, c=A_p, d=A_p, verbose=True)
with open("qcode.pkl", "wb") as f:
    pickle.dump(qcode, f)