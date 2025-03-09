import numpy as np
import ldpc.protograph as pt
from bposd.css import css_code
from bposd.stab import stab_code
import numpy as np

def mod2(A):
    """Reduce integer matrix A mod 2, ensuring it's in a NumPy-compatible format."""
    A = np.array(A, dtype=int)  # Convert A to NumPy array
    return np.remainder(A, 2)

def I(n):
    """Convenient identity of size n×n in int (mod 2)."""
    return np.eye(n, dtype=int)

class lifted_hgp(css_code):
    """
    Extended lifted hypergraph product constructor that also
    builds 4D chain-complex operators.
    """

    def __init__(self, lift_parameter, a, b=None):
        """
        Generates the 2D lifted hypergraph product of protographs a, b,
        and then constructs the 4D product operators from them.
        """
        # Store the lift parameter
        self.lift_parameter = lift_parameter
        
        # Store the protographs in GF(2) form:
        self.a = a
        if b is None:
            self.b = self.a.copy()
        else:
            self.b=b

        self.a_m, self.a_n = self.a.shape
        self.b_m, self.b_n = self.b.shape

        # ---------- 1) Usual 2D lifted HGP operators ----------
        # Hx = [ a ⊗ I_{b_n} | I_{a_m} ⊗ b^T ]
        self.hx1_proto=np.kron(self.a,I(self.b_n))
        self.hx2_proto=np.kron(I(self.a_m),self.b.T)
        self.hx_proto=pt.hstack([self.hx1_proto,self.hx2_proto])

        # Hz = [ I_{a_n} ⊗ b | a^T ⊗ I_{b_m} ]
        self.hz1_proto=np.kron(I(self.a_n),self.b)
        self.hz2_proto=np.kron(self.a.T,I(self.b_m))
        self.hz_proto=pt.hstack([self.hz1_proto,self.hz2_proto])

        # For reference, Hx has shape (a_m*b_n, ?) and Hz has shape (a_n*b_m, ?)
        # The details depend on how you stack/transpose, but that’s normal for HGP code.

        # ---------- 2) Interpret them as the 2D boundary maps ----------
        # We'll call:
        #   delta_m1_2d = hx_proto
        #   delta_0_2d  = hz_proto
        delta_m1_2d = self.hx_proto.T.to_binary(lift_parameter=self.lift_parameter)
        delta_0_2d  = self.hz_proto.to_binary(lift_parameter=self.lift_parameter)
        print(delta_m1_2d)
        print(delta_0_2d)

        # ---------- 3) Build the 4D product (Eqs. (58)–(61)) ----------
        # Let delta_0_2d be shape (k, j).  We'll do exactly what your
        # build_4d_matrices function does, using k, j for the row/column counts.
        k, j = delta_0_2d.shape
        print(f'delta_0_2d.shape: {delta_0_2d.shape}')
        print(f'delta_m1_2d.shape: {delta_m1_2d.shape}')

        # (58)  mz = δ_{-2}
        #     = [  I(k) ⊗ delta_0_2d^T ]
        #       [  delta_m1_2d ⊗ I(k) ]

        self.mz_4d = (
            np.vstack([
                np.kron(I(k), delta_0_2d.T),
                np.kron(delta_m1_2d, I(k))
            ])
        )
        print(self.mz_4d)
        print(type(self.mz_4d))

        # (59)  hz_4d = δ_{-1}^{(new)}
        #   It's a (3×2) block structure in your code:
        top_left  = np.kron(I(k), delta_m1_2d.T)
        top_right = np.zeros_like(top_left)

        mid_left  = np.kron(delta_m1_2d, I(j))
        mid_right = np.kron(I(j), delta_0_2d.T)

        bot_right = np.kron(delta_0_2d, I(k))
        bot_left = np.zeros_like(bot_right)

        self.hz_4d = (np.block([
            [top_left,  top_right],
            [mid_left,  mid_right],
            [bot_left,  bot_right]
        ]))
        print((np.block([
            [top_left,  top_right],
            [mid_left,  mid_right],
            [bot_left,  bot_right]
        ])))

        # (60)  hx_4d = δ_{0}^{(new)}
        top_left  = np.kron(delta_m1_2d, I(k))
        top_mid   = np.kron(I(j), delta_m1_2d.T)
        top_right = np.zeros_like(top_left)

        bot_left  = np.zeros_like(top_left)
        bot_mid   = np.kron(delta_0_2d, I(j))
        bot_right = np.kron(I(k), delta_0_2d.T)

        self.hx_4d = (
            np.block([
                [top_left, top_mid, top_right],
                [bot_left, bot_mid, bot_right]
            ])
        )

        # (61)  mx_4d = δ_{1}^{(new)}
        self.mx_4d = (
            np.hstack([
                np.kron(delta_0_2d, I(k)),
                np.kron(I(k), delta_m1_2d.T)  
            ])
        )

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


class bias_tailored_lifted_product(stab_code):

    def __init__(self,lift_parameter,a,b=None):

        lhgp=lifted_hgp(lift_parameter,a,b)
        
        #Hadamard rotation
        temp1=pt.hstack([pt.zeros(lhgp.hx1_proto.shape),lhgp.hz2_proto])
        temp2=pt.hstack([lhgp.hx1_proto,pt.zeros(lhgp.hz2_proto.shape)])
        self.hx_proto=pt.vstack([temp1,temp2])
        temp1=pt.hstack([lhgp.hz1_proto,pt.zeros(lhgp.hx2_proto.shape)])
        temp2=pt.hstack([pt.zeros(lhgp.hz1_proto.shape),lhgp.hx2_proto])
        self.hz_proto=pt.vstack([temp1,temp2])

        super().__init__(self.hx_proto.to_binary(lift_parameter),self.hz_proto.to_binary(lift_parameter))

    @property
    def protograph(self):
        px=pt.vstack([pt.zeros(self.hz_proto.shape),self.hx_proto])
        pz=pt.vstack([self.hz_proto,pt.zeros(self.hx_proto.shape)])
        return pt.hstack([px,pz])


