from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser
import numpy as np
from pomdp_parser import POMDPParser
from value_iteration import VI
from pomdp_util import POMDPUtil
import time

if __name__ == '__main__':
    data = np.load("rocksample_4_4.npz")
    X = data["X"]
    O = data["O"]
    Z = data["Z"]
    C = data["C"]
    P = data["P"]
    U = data["U"]
    b0 = data["b0"]
    gamma = 0.95
    n = 1
    print("Creating B_n")
    B_n = POMDPUtil.B_n(n=n, X=X)
    print("B_n created")
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    print("Creating P_b")
    P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
    print("Creating C_b")
    C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C, P=P, O=O)
    np.savez("rocksample_4_4_aggregate.npz", B_n=np.array(B_n), P_b=np.array(P_b), C_b=np.array(C_b),
             b_0_n=np.array(b_0_n), B_n_indices=np.array(B_n_indices), b_0_n_idx=b_0_n_idx)
