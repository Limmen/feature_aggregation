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
    data = np.load("rocksample_4_4_aggregate.npz")
    B_n = data["B_n"]
    B_n_indices = data["B_n_indices"]
    C_b = data["C_b"]
    P_b = data["P_b"]
    b_0_n = data["b_0_n"]
    b_0_n_idx = data["b_0_n_idx"]
    gamma = 0.95
    n = 1
    epsilon = 0.001
    start = time.time()
    mu, J = VI.vi(P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=False)
    end = time.time()
    avg_cost = POMDPUtil.policy_evaluation(L=10000, mu=mu, gamma=gamma, b0=b_0_n, P=P, C=C, Z=Z, X=X, N=100, B_n=B_n,
                                           O=O)
    print(avg_cost)
    print(end - start)
