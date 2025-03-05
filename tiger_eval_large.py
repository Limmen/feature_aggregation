from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser
import numpy as np
from value_iteration import VI
from pomdp_util import POMDPUtil
import time
from large_pomdp_parser import SparseRowO, SparseRowT
from large_pomdp_parser import load_model

if __name__ == '__main__':
    model = load_model("tiger_model.pkl")
    gamma = 0.95
    n = 4
    epsilon = 0.001
    X = list(model["state_index"].values())
    U = list(model["action_index"].values())
    O = list(model["obs_index"].values())
    b0 = list(model["start"])
    B_n = POMDPUtil.B_n(n=n, X=X)
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, model=model)
    C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, model=model)
    start = time.time()
    mu, J = VI.vi(P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=False)
    end = time.time()
    avg_cost = POMDPUtil.policy_evaluation(L=10000, mu=mu, gamma=gamma, b0=b_0_n, model=model, X=X, N=100, B_n=B_n, O=O)
    print(avg_cost)
    print(end-start)
