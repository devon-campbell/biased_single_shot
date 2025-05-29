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
    log.setLevel(logging.INFO)          # default: silent
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

def _hash_proto(P: pt.array) -> str:
    """Hash by raw bytes so identical protographs share cache entries."""
    return hex(hash(P.tobytes()))

def kron2_cached(P: pt.array, Q: pt.array, codename: str):
    name = f"k2_{_hash_proto(P)}_{_hash_proto(Q)}"
    path = f"codes/{codename}/{name}.npz"

    if os.path.exists(path):
        print(f"Loading {name}")
        return np.load(path, allow_pickle=True)["value"]       # still protograph

    K = np.kron(P, Q)        # protograph × protograph → protograph
    np.savez_compressed(path, value=K)
    return K

def kron4(A, B, C, D, codename):
    AB = kron2_cached(A, B, codename)
    CD = kron2_cached(C, D, codename)
    return np.kron(AB, CD)   # protograph result (tiny)

def cached_kron4(label, delta, I1, I2, I3, codename):
    """
    Compute or load  δ ⊗ I ⊗ I ⊗ I  (still a *protograph* — tiny, dense).

    • Stores/loads with NumPy’s compressed .npz (not sparse I/O).
    • Leaves matrix as a protograph object for later binary‑lift.
    """
    path = f"codes/{codename}"
    os.makedirs(path, exist_ok=True)
    fname = f"{path}/{label}.npz"

    if os.path.exists(fname):
        log.info("Loading %s from cache for %s", label, codename)
        return np.load(fname, allow_pickle=True)["value"]   # returns protograph

    log.info("Computing %s from scratch for %s", label, codename)
    mat = kron4(delta, I1, I2, I3, codename)               # protograph result
    np.savez_compressed(fname, value=mat)                  # save as dense .npz
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

def construct_HZ_cached(A, B, C, D, codename):
    """Return (hz_top.T, hz_bot.T) with on‑disk caching for each half."""
    def build_top():
        r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]
        return pt.vstack([
            pt.hstack([B, A, Z_cached(r, c), Z_cached(r, d)]),
            pt.hstack([C, Z_cached(r, a), A, Z_cached(r, d)]),
            pt.hstack([D, Z_cached(r, a), Z_cached(r, c), A]),
        ])

    def build_bot():
        r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]
        return pt.vstack([
            pt.hstack([Z_cached(r, b), C, B, Z_cached(r, d)]),
            pt.hstack([Z_cached(r, b), D, Z_cached(r, c), B]),
            pt.hstack([Z_cached(r, b), Z_cached(r, a), D, C]),
        ])

    hz_top = maybe_cached("hz_top", codename, build_top)
    hz_bot = maybe_cached("hz_bot", codename, build_bot)
    return hz_top.T, hz_bot.T  # same orientation as before

def construct_HX_cached(A, B, C, D, codename):
    """Return (hx_left, hx_right) with on‑disk caching for each half."""
    def build_left():
        r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]
        return pt.hstack([
            pt.vstack([C, D, Z_cached(r, c), Z_cached(r, d)]),
            pt.vstack([B, Z_cached(r, b), D, Z_cached(r, c)]),
            pt.vstack([Z_cached(r, a), B, C, Z_cached(r, b)]),
        ])

    def build_right():
        r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]
        return pt.hstack([
            pt.vstack([A, Z_cached(r, a), Z_cached(r, a), D]),
            pt.vstack([Z_cached(r, a), A, Z_cached(r, a), C]),
            pt.vstack([Z_cached(r, a), Z_cached(r, a), A, B]),
        ])

    hx_left  = maybe_cached("hx_left",  codename, build_left)
    hx_right = maybe_cached("hx_right", codename, build_right)
    return hx_left, hx_right

def construct_MZ(A, B, C, D):
    return pt.vstack([A, B, C, D]).T

def construct_MX(A, B, C, D):
    return pt.hstack([D, C, B, A])

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

def construct_4d_matrices(delta_A, delta_B, delta_C, delta_D, codename):
    A, B, C, D = get_blocks(delta_A, delta_B, delta_C, delta_D, codename)
    log.info("Got Kronecker blocks A–D")

    mz_proto = maybe_cached("mz", codename, lambda: construct_MZ(A, B, C, D))
    mx_proto = maybe_cached("mx", codename, lambda: construct_MX(A, B, C, D))
    log.info("Got metacheck matrices")

    hz1_proto, hz2_proto = construct_HZ_cached(A, B, C, D, codename)
    log.info("Got hz matrices")

    hx1_proto, hx2_proto = construct_HX_cached(A, B, C, D, codename)
    log.info("Got hx matrices")

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