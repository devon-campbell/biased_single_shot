import numpy as np
from ldpc import protograph as pt
from bposd.css import css_code
import scipy.sparse as sp
import logging
import time

# ----------------------------------------------------------------------
# logger setup (each file that imports this module gets the same logger)
log = logging.getLogger(__name__)
if not log.handlers:                       # avoid duplicate handlers
    h = logging.StreamHandler()
    fmt = "[%(levelname)s] %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    log.addHandler(h)
    log.setLevel(logging.WARNING)          # default: silent
# ----------------------------------------------------------------------

def kron4(A, B, C, D):
    """Return the Kronecker product A ⊗ B ⊗ C ⊗ D."""
    log.debug("kron4: operands shapes %s × %s × %s × %s", 
              A.shape, B.shape, C.shape, D.shape)
    return np.kron(np.kron(np.kron(A, B), C), D)

def eye_pt(n):
    """Protograph identity matrix of size n×n."""
    return pt.identity(n)         # <- returns ldpc.protograph.array

def get_identities(delta_A, delta_B, delta_C, delta_D):
    """Return protograph‑type identity matrices that match the codomain size
       of each δ‑matrix.
    """
    return (
        eye_pt(delta_A.shape[1]),
        eye_pt(delta_B.shape[1]),
        eye_pt(delta_C.shape[1]),
        eye_pt(delta_D.shape[1]),
    )

def verify_delta_condition(delta_i, delta_i_minus_1):
    product = (delta_i @ delta_i_minus_1) & 1  # uint8 matrices
    ok = product.nnz == 0
    log.debug("verify_delta_condition: nnz=%d → %s",
              product.nnz, "OK" if ok else "FAIL")
    return ok

def get_blocks(delta_A, delta_B, delta_C, delta_D):
    t0 = time.perf_counter()
    I_f, I_h, I_j, I_l = get_identities(delta_A, delta_B, delta_C, delta_D)
    A = kron4(delta_A, I_h, I_j, I_l)
    B = kron4(I_f, delta_B, I_j, I_l)
    C = kron4(I_f, I_h, delta_C, I_l)
    D = kron4(I_f, I_h, I_j, delta_D)
    log.debug("get_blocks: built A,B,C,D in %.3f s", time.perf_counter()-t0)
    return A, B, C, D

# Module-level zero matrix cache
_zero_cache = {}

def Z_cached(rows, cols):
    """Cached zero protograph matrices."""
    key = (rows, cols)
    if key not in _zero_cache:
        _zero_cache[key] = pt.zeros(key)
        log.debug("Z_cached: created zeros(%d, %d)", rows, cols)
    return _zero_cache[key]

def construct_HZ(A, B, C, D):
    """Construct H_Z as a 6×4 protograph block matrix."""
    t0 = time.perf_counter()

    # Consistent shape unpacking
    r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]

    rows = [
        [B, A, Z_cached(r, c), Z_cached(r, d)],
        [C, Z_cached(r, a), A, Z_cached(r, d)],
        [D, Z_cached(r, a), Z_cached(r, c), A],
        [Z_cached(r, b), C, B, Z_cached(r, d)],
        [Z_cached(r, b), D, Z_cached(r, c), B],
        [Z_cached(r, b), Z_cached(r, a), D, C]
    ]

    hz1_proto = pt.vstack([pt.hstack(row) for row in rows[:3]]).T
    hz2_proto = pt.vstack([pt.hstack(row) for row in rows[3:]]).T

    log.debug("construct_HZ completed in %.3f s", time.perf_counter() - t0)

    return hz1_proto, hz2_proto

def construct_HX(A, B, C, D):
    """Construct H_X as a 4×6 protograph block matrix."""
    t0 = time.perf_counter()

    # Consistent shape unpacking
    r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]

    cols = [
        pt.vstack([C, D, Z_cached(r, c), Z_cached(r, d)]),
        pt.vstack([B, Z_cached(r, b), D, Z_cached(r, c)]),
        pt.vstack([Z_cached(r, a), B, C, Z_cached(r, b)]),
        pt.vstack([A, Z_cached(r, a), Z_cached(r, a), D]),
        pt.vstack([Z_cached(r, a), A, Z_cached(r, a), C]),
        pt.vstack([Z_cached(r, a), Z_cached(r, a), A, B])
    ]

    hx1_proto = pt.hstack(cols[:3])
    hx2_proto = pt.hstack(cols[3:])

    log.debug("construct_HX completed in %.3f s", time.perf_counter() - t0)

    return hx1_proto, hx2_proto

def construct_MZ(A, B, C, D):
    return pt.array(pt.vstack([A, B, C, D]).T)

def construct_MX(A, B, C, D):
    return pt.array(pt.hstack([D, C, B, A]))

def verify_4d_matrices(mz, hz, hx, mx):
    deltas = verify_delta_condition(hz.T, mz.T) and verify_delta_condition(hx, hz.T) and verify_delta_condition(mx, hx)
    h1, h2 = verify_delta_condition(hx, hz.T), verify_delta_condition(hz, hx.T)
    return deltas and h1 and h2    

def construct_4d_matrices(delta_A, delta_B, delta_C, delta_D):
    A, B, C, D = get_blocks(delta_A, delta_B, delta_C, delta_D)
    log.info("\tgot blocks")
    mz_proto = construct_MZ(A, B, C, D)
    log.info("\tmz done")
    hz1_proto, hz2_proto = construct_HZ(A, B, C, D)
    log.info("\thz done")
    hx1_proto, hx2_proto = construct_HX(A, B, C, D)
    log.info("\thx done")
    mx_proto = construct_MX(A, B, C, D)
    log.info("\tmx done")
    
    return mz_proto, hz1_proto, hz2_proto, hx1_proto, hx2_proto, mx_proto

class lifted_hgp_4d(css_code):
    """
    Extended lifted hypergraph product constructor that also
    builds 4D chain-complex operators.
    """

    def __init__(self, lift_parameter, a, b, c, d, *, verbose=False):
        """
        Generates the 2D lifted hypergraph product of protographs a, b,
        and then constructs the 4D product operators from them.
        """
        # ----------------------------------------------------------
        # handle verbosity
        if verbose:
            level = logging.DEBUG if verbose == "debug" else logging.INFO
            log.setLevel(level)
        log.info("lifted_hgp_4d: starting construction (L=%s)", lift_parameter)
        t_global = time.perf_counter()
        # ----------------------------------------------------------
        # Store the lift parameter
        self.lift_parameter = lift_parameter

        log.info("begin construct_4d_matrices:")
        t0 = time.perf_counter()
        self.mz_proto, self.hz1_proto, self.hz2_proto, \
            self.hx1_proto, self.hx2_proto, self.mx_proto = \
                construct_4d_matrices(a, b, c, d)
        log.info("  protograph assembly done in %.2f s", time.perf_counter()-t0)


        self.hz_proto=pt.hstack([self.hz1_proto, self.hz2_proto])
        t0 = time.perf_counter()
        self.hz = self.hz_proto.to_binary(lift_parameter)
        log.info("  hz binary lifting done in %.2f s", time.perf_counter()-t0)

        self.hx_proto=pt.hstack([self.hx1_proto, self.hx2_proto])
        t0 = time.perf_counter()
        self.hx = self.hx_proto.to_binary(lift_parameter)
        log.info("  hx binary lifting done in %.2f s", time.perf_counter()-t0)

        t0 = time.perf_counter()
        self.mz = self.mz_proto.to_binary(lift_parameter)        
        self.mx = self.mx_proto.to_binary(lift_parameter)
        log.info("  metacheck binary lifting done in %.2f s", time.perf_counter()-t0)
        log.info("lifted_hgp_4d: finished in %.2f s", time.perf_counter()-t_global)
    
        super().__init__(self.hx_proto.to_binary(lift_parameter),self.hz_proto.to_binary(lift_parameter))

    @property
    def protograph(self):
        px=pt.vstack([pt.zeros(self.hz_proto.shape),self.hx_proto])
        pz=pt.vstack([self.hz_proto,pt.zeros(self.hx_proto.shape)])
        return pt.hstack([px,pz])

    @property
    def hx1(self):
        return self.hx1_proto.to_binary(self.lift_parameter)
    @property
    def hx2(self):
        return self.hx2_proto.to_binary(self.lift_parameter)
    @property
    def hz1(self):
        return self.hz1_proto.to_binary(self.lift_parameter)
    @property
    def hz2(self):
        return self.hz2_proto.to_binary(self.lift_parameter)