import numpy as np
from ldpc import protograph as pt
from bposd.css import css_code
from bposd.stab import stab_code         # non-CSS stabiliser base class
import scipy.sparse as sp
import logging
import time
import os

# ----------------------------------------------------------------------
# logger setup (each file that imports this module gets the same logger)
log = logging.getLogger(__name__)
if not log.handlers:                       # avoid duplicate handlers
    h = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    log.addHandler(h)
    log.setLevel(logging.WARNING)          # default: silent
# ----------------------------------------------------------------------

# Module-level zero matrix cache
_eye_cache = {}
_zero_cache = {}

def I_cached(n):
    """Cached I protograph matrices."""
    if n not in _eye_cache:
        _eye_cache[n] = pt.identity(n)   
        log.debug("I_cached: created identity %d", n)
    return _eye_cache[n]

def Z_cached(rows, cols):
    """Cached zero protograph matrices."""
    key = (rows, cols)
    if key not in _zero_cache:
        _zero_cache[key] = pt.zeros(key)
        log.debug("Z_cached: created zeros(%d, %d)", rows, cols)
    return _zero_cache[key]

def save_matrix(proto, key, codename):
    path = f"codes/{codename}"
    os.makedirs(path, exist_ok=True)  # Create dir if needed
    np.savez_compressed(f"{path}/{key}.npz", value=proto)

def load_matrix(label, codename):
    return np.load(f"codes/{codename}/{label}.npz", allow_pickle=True)['value']

def cached_kron4(label, delta, I1, I2, I3, codename):
    """Compute or load Kronecker product for a given delta and 3 identity blocks."""
    fname = f"codes/{codename}/{label}.npz"
    if os.path.exists(fname):
        log.info(f"Loading {label} from cache")
        return load_matrix(label, codename)
    else:
        log.info(f"Computing {label} from scratch")
        mat = kron4(delta, I1, I2, I3)
        save_matrix(mat, label, codename)
        return mat


def get_identities(delta_A, delta_B, delta_C, delta_D):
    """Return protograph‑type identity matrices that match the codomain size
       of each δ‑matrix.
    """
    return (
        I_cached(delta_A.shape[1]),
        I_cached(delta_B.shape[1]),
        I_cached(delta_C.shape[1]),
        I_cached(delta_D.shape[1]),
    )

def get_blocks(delta_A, delta_B, delta_C, delta_D, codename):
    """
    Computes or loads the 4 Kronecker blocks used in the 4D lifted HGP code.
    """
    t0 = time.perf_counter()
    I_f, I_h, I_j, I_l = get_identities(delta_A, delta_B, delta_C, delta_D)
    log.info("Got identity matrices")

    A = cached_kron4("A", delta_A, I_h, I_j, I_l, codename)
    B = cached_kron4("B", delta_B, I_f, I_j, I_l, codename)
    C = cached_kron4("C", delta_C, I_f, I_h, I_l, codename)
    D = cached_kron4("D", delta_D, I_f, I_h, I_j, codename)

    log.info("Constructed A, B, C, D in %.3f seconds", time.perf_counter() - t0)
    return A, B, C, D

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

    log.info('got hz rows')
    hz1_proto = pt.vstack([pt.hstack(row) for row in rows[:3]]).T
    hz2_proto = pt.vstack([pt.hstack(row) for row in rows[3:]]).T
    log.info('stacked hz rows')
    
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
    log.info('got hz rows')
    hx1_proto = pt.hstack(cols[:3])
    hx2_proto = pt.hstack(cols[3:])
    log.info('stacks hz rows')

    log.debug("construct_HX completed in %.3f s", time.perf_counter() - t0)

    return hx1_proto, hx2_proto

def construct_MZ(A, B, C, D):
    return pt.array(pt.vstack([A, B, C, D]).T)

def construct_MX(A, B, C, D):
    return pt.array(pt.hstack([D, C, B, A]))

def maybe_cached(label, codename, compute_fn):
    """Loads matrix from cache if available, otherwise computes and saves it."""
    path = f"codes/{codename}/{label}.npz"
    if os.path.exists(path):
        log.info(f"Loading {label} from cache")
        return load_matrix(label, codename)
    else:
        log.info(f"Computing {label} from scratch")
        result = compute_fn()
        save_matrix(result, label, codename)
        return result

def construct_4d_matrices(delta_A, delta_B, delta_C, delta_D, codename="code"):
    A, B, C, D = get_blocks(delta_A, delta_B, delta_C, delta_D, codename)
    log.info("Got Kronecker blocks A–D")

    mz_proto = maybe_cached("mz", codename, lambda: construct_MZ(A, B, C, D))
    hz1_proto = maybe_cached("hz1", codename, lambda: construct_HZ(A, B, C, D)[0])
    hz2_proto = maybe_cached("hz2", codename, lambda: construct_HZ(A, B, C, D)[1])
    hx1_proto = maybe_cached("hx1", codename, lambda: construct_HX(A, B, C, D)[0])
    hx2_proto = maybe_cached("hx2", codename, lambda: construct_HX(A, B, C, D)[1])
    mx_proto  = maybe_cached("mx", codename, lambda: construct_MX(A, B, C, D))

    log.info("Constructed all 4D matrices")

    return mz_proto, hz1_proto, hz2_proto, hx1_proto, hx2_proto, mx_proto

def split_cols(mat):
    mid = mat.shape[1] // 2
    return mat[:, :mid], mat[:, mid:]

def modmult(m1, m2, n = 2):
    return (m1 @ m2).toarray() % 2

def modmult(m1, m2, n = 2):
    return (m1 @ m2).toarray() % 2

def H_commute(hx, hz):
    """Return True⇔ all stabilisers commute."""
    combo = (modmult(hz, hx.T) + modmult(hx, hz.T)) % 2
    return combo.sum() == 0

def MHHM_commute(mz, hz, hx, mx):
    combo = (mx @ hx @ hz.T @ mz.T).toarray() % 2
    return combo.sum() == 0

def split_cols(mat):
    """Split a protograph matrix into left and right halves along columns."""
    mid = mat.shape[1] // 2
    return mat[:, :mid], mat[:, mid:]

def save_alist(name, mat, j=None, k=None):
    '''
    Function converts parity check matrix into the format required for the RN decoder
    '''
    H=np.copy(mat)
    H=H.T

    if j is None:
        j=int(max(H.sum(axis=0)))

    if k is None:
        k=int(max(H.sum(axis=1)))

    m, n = H.shape # rows, cols
    f = open(name, 'w')
    print(n, m, file=f)
    print(j, k, file=f)

    for col in range(n):
        print( int(H[:, col].sum()), end=" ", file=f)
    print(file=f)
    for row in range(m):
        print( int(H[row, :].sum()), end=" ", file=f)
    print(file=f)

    for col in range(n):
        for row in range(m):
            if H[row, col]:
                print( row+1, end=" ", file=f)
        print(file=f)

    for row in range(m):
        for col in range(n):
            if H[row, col]:
                print(col+1, end=" ", file=f)
        print(file=f)
    f.close()

class lifted_hgp_4d(css_code):
    """
    Extended lifted hypergraph product constructor that also
    builds 4D chain-complex operators.
    """

    def __init__(self, lift_parameter, a, b, c, d, codename, *, verbose=False):
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
                construct_4d_matrices(a, b, c, d, codename)
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
        self.mz = sp.csr_matrix(self.mz_proto.to_binary(lift_parameter))        
        self.mx = sp.csr_matrix(self.mx_proto.to_binary(lift_parameter))
        log.info("  metacheck binary lifting done in %.2f s", time.perf_counter()-t0)
        log.info("lifted_hgp_4d: finished in %.2f s", time.perf_counter()-t_global)
        print('hx bin shape', self.hx_proto.to_binary(lift_parameter).shape)
        print('hz bin shape', self.hz_proto.to_binary(lift_parameter).shape)
    
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

# ────────────────────────────────────────────────────────────────────────────
#  Bias-tailored 4-D Lifted HGP code with XZZX-style Hadamard twist
# ────────────────────────────────────────────────────────────────────────────

class bias_tailored_lhp_4d(stab_code):
    """
    Construct a *bias-tailored* 4-dimensional lifted-hypergraph-product code
    by applying a block-Hadamard twist (X↔Z swap) to half of the qubits
    of an untwisted lifted_hgp_4d instance.  This is the 4-D analogue of
    the XZZX-twisted toric code.
    """

    def __init__(self, L, a, b, c, d, *,  verbose=False):
        """
        Parameters
        ----------
        L : int
            Circulant lift parameter.
        a,b,c,d : pt.array
            Classical protographs used to build the 4-D lifted product.
        verbose : bool or {"debug"}
            Forwarded to the inner lifted_hgp_4d constructor.
        """

        # ── 1. Build the untwisted 4-D code ────────────────────────────────
        from lifted_hgp_4d import lifted_hgp_4d   # avoid circular import
        lhgp = lifted_hgp_4d(
            L, a, b, c, d, verbose=verbose
        )

        # Ensure equal-sized halves for the Hadamard swap
        n1 = lhgp.hx1_proto.shape[1]
        n2 = lhgp.hx2_proto.shape[1]
        assert n1 == n2, "Hadamard twist requires N/2-N/2 partition"

        # ── 2. Hadamard-swap second half:  hx' , hz'  ─────────────────────
        hx1, hx2 = split_cols(lhgp.hx_proto)
        hz1, hz2 = split_cols(lhgp.hz_proto)
        
        # Hadamard twist: swap roles on second half
        self.hx_proto = pt.hstack([ hx1, hz2 ])  # first half unchanged, second half X←→Z
        self.hz_proto = pt.hstack([ hz1, hx2 ])
  
        # 3. Set the metacheck matrices
        self.mz_proto = lhgp.mz_proto
        self.mx_proto = lhgp.mx_proto

        # ── 4. Lift to binary & build the stabiliser code ────────────────
        t0 = time.perf_counter()
        self.hx = sp.csr_matrix(self.hx_proto.to_binary(L))
        self.hz = sp.csr_matrix(self.hz_proto.to_binary(L))

        self.mz = sp.csr_matrix(self.mz_proto.to_binary(L))
        self.mx = sp.csr_matrix(self.mx_proto.to_binary(L))

        print(f'(hz @ hx.T) + (hx @ hz.T) == 0?: {H_commute(self.hx, self.hz)}')
        print(f'(mx @ hx @ hz.T @ mz.T) == 0?: {MHHM_commute(self.mz, self.hz, self.hx, self.mx)}')

        log.info("bias_tailored_lhp_4d: binary lifting done in %.2f s",
                 time.perf_counter() - t0)

        super().__init__(self.hx, self.hz)    # non-CSS base class

    # ── 5. Convenience helpers ────────────────────────────────────────────
    @property
    def protograph(self):
        px = pt.vstack([ pt.zeros(self.hz_proto.shape), self.hx_proto ])
        pz = pt.vstack([ self.hz_proto, pt.zeros(self.hx_proto.shape) ])
        return pt.hstack([ px, pz ])

    @property
    def n_phys(self):          # number of physical qubits
        return self.hx.shape[1]

    # Expose lower/upper halves if needed
    @property
    def hx1_proto(self): return self.hx_proto[: , : self.hx_proto.shape[1]//2]
    @property
    def hx2_proto(self): return self.hx_proto[: ,  self.hx_proto.shape[1]//2 :]

    @property
    def hz1_proto(self): return self.hz_proto[: , : self.hz_proto.shape[1]//2]
    @property
    def hz2_proto(self): return self.hz_proto[: ,  self.hz_proto.shape[1]//2 :]