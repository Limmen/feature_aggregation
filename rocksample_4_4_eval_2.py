from value_iteration import VI
from vi_2 import vi
from vi_3 import vi as vi_3
from pomdp_util import POMDPUtil
import time
from large_pomdp_parser import SparseRowO, SparseRowT, get_next_state_obs_pairs
from large_pomdp_parser import load_model
import numpy as np

if __name__ == '__main__':
    model = load_model("rocksample_4_4.pkl")
    gamma = 0.95
    n = 3
    epsilon = 0.001
    X = list(model["state_index"].values())
    U = list(model["action_index"].values())
    O = list(model["obs_index"].values())
    b0 = list(model["start"])
    # reachable_beliefs = POMDPUtil.enumerate_reachable_beliefs(model=model, b0=b0, X=X, U=U, T=3)
    # print(len(reachable_beliefs))
    B_n = POMDPUtil.B_n(n=n, X=X)
    # B_n = POMDPUtil.sample_beliefs_halton(num_beliefs=20, num_states=len(X))
    # len(B_n)
    # annoy_index = POMDPUtil.build_annoy_index(B_n=B_n, num_trees=5)
    annoy_index= None
    b_to_index = {}
    for i, b in enumerate(B_n):
        if not isinstance(b, tuple):
            b = tuple(b)
            B_n[i] = b
        b_to_index[b] = i
    cost_xu = POMDPUtil.precompute_cost(U=U, X=X, model=model)
    start = time.time()
    mu, J = vi(B_n=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, cost_xu=cost_xu,
               annoy_index=annoy_index, b_to_index=b_to_index, n=n)
    # mu, J = vi_3(B_list=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, annoy_index=annoy_index)
    end = time.time()
    print(end-start)
    avg_cost = POMDPUtil.policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0, model=model, X=X, N=100, B_n=B_n,
                                           annoy_index=annoy_index, b_to_index=b_to_index, n=n)
    print(avg_cost)
