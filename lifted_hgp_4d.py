import numpy as np
from ldpc import protograph as pt
from bposd.css import css_code

def kron4(A, B, C, D):
    """Return the Kronecker product A ⊗ B ⊗ C ⊗ D."""
    return np.kron(np.kron(np.kron(A, B), C), D)

def get_identities(delta_A, delta_B, delta_C, delta_D):
    """Return identity matrices for the codomains of delta matrices."""
    return (
        np.eye(delta_A.shape[1]),
        np.eye(delta_B.shape[1]),
        np.eye(delta_C.shape[1]),
        np.eye(delta_D.shape[1])
    )

def get_blocks(delta_A, delta_B, delta_C, delta_D):
    I_f, I_h, I_j, I_l = get_identities(delta_A, delta_B, delta_C, delta_D)
    A = kron4(delta_A, I_h, I_j, I_l)
    B = kron4(I_f, delta_B, I_j, I_l)
    C = kron4(I_f, I_h, delta_C, I_l)
    D = kron4(I_f, I_h, I_j, delta_D)
    return A, B, C, D

def construct_HZ(delta_A, delta_B, delta_C, delta_D):
    """Construct H_Z as a 6×4 protograph block matrix following the 4D homological code definition."""
    A, B, C, D = get_blocks(delta_A, delta_B, delta_C, delta_D)
    r, a, b, c, d = B.shape[0], A.shape[1], B.shape[1], C.shape[1], D.shape[1]

    def Z(rows, cols):
        return pt.zeros((rows, cols))

    rows = [
        [B, A,      Z(r, c), Z(r, d)],
        [C, Z(r, a), A,      Z(r, d)],
        [D, Z(r, a), Z(r, c), A],
        [Z(r, b), C, B,      Z(r, d)],
        [Z(r, b), D, Z(r, c), B],
        [Z(r, b), Z(r, a), D, C]
    ]
    hz1_proto = pt.vstack([pt.hstack(row) for row in rows[:3]]).T
    hz2_proto = pt.vstack([pt.hstack(row) for row in rows[3:]]).T

    return hz1_proto, hz2_proto

def construct_HX(delta_A, delta_B, delta_C, delta_D):
    """Construct H_X as a 4×6 protograph block matrix following the 4D homological code definition."""
    A, B, C, D = get_blocks(delta_A, delta_B, delta_C, delta_D)

    # Shape unpacking
    rA, cA = A.shape
    rB, cB = B.shape
    rC, cC = C.shape
    rD, cD = D.shape

    def Z(rows, cols):
        return pt.zeros((rows, cols))

    cols = [
        pt.vstack([C, D, Z(rD, cC), Z(rD, cD)]),
        pt.vstack([B, Z(rB, cB), D, Z(rD, cC)]),
        pt.vstack([Z(rC, cA), B, C, Z(rD, cB)]),
        pt.vstack([A, Z(rA, cA), Z(rC, cA), D]),
        pt.vstack([Z(rC, cA), A, Z(rC, cA), C]),
        pt.vstack([Z(rC, cA), Z(rA, cA), A, B])
    ]

    hx1_proto = pt.hstack(cols[:3])
    hx2_proto = pt.hstack(cols[3:])

    return hx1_proto, hx2_proto

def construct_MZ(delta_A, delta_B, delta_C, delta_D):
    I_f, I_h, I_j, I_l = get_identities(delta_A, delta_B, delta_C, delta_D)

    rows = [
        kron4(delta_A, I_h, I_j, I_l),
        kron4(I_f, delta_B, I_j, I_l),
        kron4(I_f, I_h, delta_C, I_l),
        kron4(I_f, I_h, I_j, delta_D),
    ]
    return pt.array(pt.vstack(rows).T)

def construct_MX(delta_A, delta_B, delta_C, delta_D):
    I_f, I_h, I_j, I_l = get_identities(delta_A, delta_B, delta_C, delta_D)

    blocks = [
        kron4(I_f, I_h, I_j, delta_D),
        kron4(I_f, I_h, delta_C, I_l),
        kron4(I_f, delta_B, I_j, I_l),
        kron4(delta_A, I_h, I_j, I_l),
    ]
    return pt.array(pt.hstack(blocks))

def verify_delta_condition(delta_i, delta_i_minus_1):
    """
    Verify that δ_i * δ_{i-1} = 0 mod 2.
    """
    product = np.mod(np.dot(delta_i, delta_i_minus_1), 2)  # Compute δ_i * δ_{i-1} mod 2
    return np.all(product == 0)

def verify_4d_matrices(mz, hz, hx, mx):
    deltas = verify_delta_condition(hz.T, mz.T) and verify_delta_condition(hx, hz.T) and verify_delta_condition(mx, hx)
    h1, h2 = verify_delta_condition(hx, hz.T), verify_delta_condition(hz, hx.T)
    return deltas and h1 and h2    

def construct_4d_matrices(delta_A, delta_B, delta_C, delta_D):
    mz_proto = construct_MZ(delta_A, delta_B, delta_C, delta_D)
    hz1_proto, hz2_proto = construct_HZ(delta_A, delta_B, delta_C, delta_D)
    hx1_proto, hx2_proto = construct_HX(delta_A, delta_B, delta_C, delta_D)
    mx_proto = construct_MX(delta_A, delta_B, delta_C, delta_D)
    
    return mz_proto, hz1_proto, hz2_proto, hx1_proto, hx2_proto, mx_proto

class lifted_hgp_4d(css_code):
    """
    Extended lifted hypergraph product constructor that also
    builds 4D chain-complex operators.
    """

    def __init__(self, lift_parameter, a, b, c, d):
        """
        Generates the 2D lifted hypergraph product of protographs a, b,
        and then constructs the 4D product operators from them.
        """
        # Store the lift parameter
        self.lift_parameter = lift_parameter
        
        # Store the protographs in GF(2) form:
        self.a=a
        self.b=b
        self.c=c
        self.d=d

        self.mz_proto, self.hz1_proto, self.hz2_proto, self.hx1_proto, self.hx2_proto, self.mx_proto = construct_4d_matrices(a, b, c, d)

        self.hz_proto=pt.hstack([self.hz1_proto, self.hz2_proto])
        self.hz = self.hz_proto.to_binary(lift_parameter)
        
        self.hx_proto=pt.hstack([self.hx1_proto, self.hx2_proto])
        self.hx = self.hx_proto.to_binary(lift_parameter)
        
        self.mz = self.mz_proto.to_binary(lift_parameter)        
        self.mx = self.mx_proto.to_binary(lift_parameter)
    
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