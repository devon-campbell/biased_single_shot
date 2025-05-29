import logging
from functools import reduce
from z3 import Optimize, Bool, Xor, Sum, If, Or, is_true, sat, Solver
from datetime import datetime
import numpy as np

def compute_min_hamming_distance(G):
    start_time = datetime.now()
    print(f"Started computation at {start_time}")

    # Initialize Z3 optimizer
    optimizer = Optimize()

    # Number of rows in generator matrix G (number of codewords)
    k = len(G)
    n = len(G[0])  # Length of each codeword

    print(f"Generator matrix has {k} rows and {n} columns.")

    # Define Boolean variables for each row of G
    b = [Bool(f'b_{i}') for i in range(k)]

    # Add constraint: At least one row is selected (non-zero codeword)
    optimizer.add(Or(b))

    # Helper function for XOR over multiple terms
    def safe_xor(terms):
        if not len(terms):
            return False  # Represents zero in Boolean logic
        elif len(terms) == 1:
            return terms[0]
        else:
            return reduce(lambda x, y: Xor(x, y), terms)

    # Precompute codeword structure
    codeword = []
    for j in range(n):
        xor_result = safe_xor([b[i] for i in range(k) if G[i][j] == 1])
        codeword.append(xor_result)
        # if not j%15:
        #     print(f"Computed codeword bit {j}: {xor_result}")

    # Define Hamming weight
    hamming_weight = Sum([If(bit, 1, 0) for bit in codeword])

    # Ensure non-zero Hamming weight (exclude zero codeword)
    optimizer.add(hamming_weight > 0)

    # Set the objective to minimize the Hamming weight
    optimizer.minimize(hamming_weight)

    print('Finding Hamming distance...')
    if optimizer.check() == sat:
        model = optimizer.model()
        min_distance = model.evaluate(hamming_weight).as_long()
        selected_rows = [i for i in range(k) if is_true(model.evaluate(b[i]))]
        print(f"Minimum non-zero Hamming distance found: {min_distance}")

        # Compute the codeword values
        computed_codeword = [is_true(model.evaluate(bit)) for bit in codeword]
        binary_codeword = [1 if bit else 0 for bit in computed_codeword]

        # # Log the contributing rows
        print(f"Contributing rows from G: {selected_rows}")
        # for idx in selected_rows:
        #     print(f"Row {idx}: {G[idx]}")

        # Log the resulting codeword
        print(f"XOR Result (codeword): {binary_codeword}")

        end_time = datetime.now()
        print(f"Ended computation at {end_time}")
        print(f"Total computation time: {end_time - start_time}")

        return min_distance
    else:
        print("No valid codeword found.")
        end_time = datetime.now()
        print(f"Ended computation at {end_time}")
        print(f"Total computation time: {end_time - start_time}")
        return "No valid codeword found"

def gf2_add_row(r1, r2):
    return [a ^ b for a, b in zip(r1, r2)]

def gf2_rref(matrix):
    M = [row[:] for row in matrix]
    rows = len(M)
    cols = len(M[0]) if rows > 0 else 0

    pivot_positions = []
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        # Find pivot
        pivot = -1
        for r in range(pivot_row, rows):
            if M[r][col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue

        # Swap to put pivot in place
        if pivot != pivot_row:
            M[pivot_row], M[pivot] = M[pivot], M[pivot_row]

        # Eliminate down the column
        for r in range(rows):
            if r != pivot_row and M[r][col] == 1:
                M[r] = gf2_add_row(M[r], M[pivot_row])

        pivot_positions.append(col)
        pivot_row += 1
    return M, pivot_positions

def convert_stabilizer_matrix_to_generator_matrix(H):
    R, pivots = gf2_rref(H)
    n = len(H[0])
    r = len(pivots)
    k = n - r
    if k <= 0:
        return []

    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]

    G = []
    for free_col in free_cols:
        vec = [0]*n
        vec[free_col] = 1
        # Solve pivot vars
        for row_idx, pcol in enumerate(pivots):
            rhs = 0
            for c, val in enumerate(R[row_idx]):
                if c != pcol and val == 1:
                    rhs ^= vec[c]
            vec[pcol] = rhs
        G.append(vec)

    return G

