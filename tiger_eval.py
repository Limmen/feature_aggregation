from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser
import numpy as np
from pomdp_parser import POMDPParser
from value_iteration import VI
from pomdp_util import POMDPUtil
import time

if __name__ == '__main__':
    alpha_vectors = PomdpSolveParser.parse_alpha_vectors(file_path="tiger.alpha")
    b0 = [0.5, 0.5]
    val = np.min([np.dot(b0, list(-np.array(alpha[1]))) for alpha in alpha_vectors])
    print(val)
    pomdp = POMDPParser.parse_pomdp(file_path="tiger.pomdp")
    X = pomdp.X
    O = pomdp.O
    Z = pomdp.Z
    C = list(-np.array(pomdp.R))
    P = pomdp.T
    U = pomdp.A
    gamma = 0.95
    n = 4
    epsilon = 0.001
    B_n = POMDPUtil.B_n(n=n, X=X)
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, O=O, P=P, Z=Z)
    C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, C=C, P=P, O=O)
    start = time.time()
    mu, J = VI.vi(P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=False)
    end = time.time()
    avg_cost = POMDPUtil.policy_evaluation(L=10000, mu=mu, gamma=gamma, b0=b_0_n, P=P, C=C, Z=Z, X=X, N=100, B_n=B_n, O=O)
    if avg_cost < val:
        avg_cost = val
    print(avg_cost)
    print(end-start)
    # print(J[b_0_n_idx])
