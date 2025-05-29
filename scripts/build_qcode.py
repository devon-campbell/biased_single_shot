import os
import joblib
import numpy as np
import argparse
from ldpc import protograph as pt
from mm_qc_pega.graphs import QC_tanner_graph
from lifted_hgp_4d import lifted_hgp_4d

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

def pad_protograph_vertically(P: pt.array) -> pt.array:
    """
    Make a square protograph by adding all‑zero rows until m == n.
    Works for QC or generic protographs.
    """
    m, n = P.shape
    if m == n:
        return P            # already square
    if m > n:
        raise ValueError("Protograph has more rows than columns; can’t pad vertically.")

    pad_rows = n - m
    zero_row = pt.zeros((1, n))          # a single all‑zero protograph row
    P_padded = pt.vstack([P] + [zero_row] * pad_rows)
    return P_padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codename", help="Name of the PEG-generated .qc file (without extension)")
    args = parser.parse_args()

    codename = args.codename
    G = QC_tanner_graph.read(f"../mm_qc_pega/pegs/{codename}.qc")
    A_p = qc_proto_to_pt(G.proto, N=G.N)
    A_sq = pad_protograph_vertically(A_p)

    qcode = lifted_hgp_4d(G.N, A_sq, A_sq, A_sq, A_sq, codename)
    os.makedirs(f"../codes/{codename}", exist_ok=True)
    joblib.dump(qcode, f"../codes/{codename}/qcode.pkl.z", compress=3)

if __name__ == "__main__":
    main()