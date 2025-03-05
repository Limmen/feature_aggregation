from value_iteration import VI
from vi_2 import on_the_fly_value_iteration
from pomdp_util import POMDPUtil
import time
from large_pomdp_parser import SparseRowO, SparseRowT
from large_pomdp_parser import load_model
import numpy as np

if __name__ == '__main__':
    model = load_model("rocksample_4_4.pkl")
    gamma = 0.95
    n = 2
    epsilon = 0.001
    X = list(model["state_index"].values())
    U = list(model["action_index"].values())
    O = list(model["obs_index"].values())
    b0 = list(model["start"])
    # reachable_beliefs = POMDPUtil.enumerate_reachable_beliefs(model=model, b0=b0, X=X, U=U, T=3)
    # print(len(reachable_beliefs))
    B_n = POMDPUtil.B_n(n=n, X=X)
    on_the_fly_value_iteration(B_list=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model)
    # data = np.load("B_n.npz")
    # B_n = data["B_n"].tolist()
    # B_n = POMDPUtil.sample_beliefs_halton(num_beliefs=400, num_states=len(X))
    # b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    # B_n_indices = [i for i in range(len(B_n))]
    # b_0_n_idx = B_n.index(b_0_n)
    # print("Creating P_b")
    # P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, model=model)
    # print("Creating C_b")
    # C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, model=model)
    # start = time.time()
    # print("Starting VI")
    # mu, J = VI.vi(P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=False)
    # print("VI finished")
    # end = time.time()
    # avg_cost = POMDPUtil.policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0, model=model, X=X, N=100, B_n=B_n)
    # print(avg_cost)
    # print(end-start)
