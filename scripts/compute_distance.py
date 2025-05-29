import sys
import joblib
from distance import compute_min_hamming_distance


codename = 'peg_6_4_3_0-1'
qcode = joblib.load(f"../codes/{codename}/qcode.pkl.z")

with open(f"../codes/{codename}/distance_log.txt", "w") as f:
    sys.stdout = f
    sys.stderr = f

    print("Computing Hamming distance for Hx")
    compute_min_hamming_distance(qcode.hx.toarray())
    
    print("Computing Hamming distance for Mx")
    compute_min_hamming_distance(qcode.mx.toarray())
