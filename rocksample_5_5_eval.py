from value_iteration import VI
from pomdp_util import POMDPUtil
import time
from large_pomdp_parser import SparseRowO, SparseRowT
from large_pomdp_parser import load_model

if __name__ == '__main__':
    model = load_model("rocksample_5_5.pkl")
    gamma = 0.95
    n = 1
    epsilon = 0.001
    X = list(model["state_index"].values())
    U = list(model["action_index"].values())
    O = list(model["obs_index"].values())
    b0 = list(model["start"])
    B_n = POMDPUtil.B_n(n=n, X=X)
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    print("Creating P_b")
    P_b = POMDPUtil.P_b(B_n=B_n, X=X, U=U, model=model)
    print("Creating C_b")
    C_b = POMDPUtil.C_b(B_n=B_n, X=X, U=U, model=model)
    start = time.time()
    print("Starting VI")
    mu, J = VI.vi(P=P_b, epsilon=epsilon, gamma=gamma, C=C_b, X=B_n_indices, U=U, verbose=False)
    print("VI finished")
    end = time.time()
    avg_cost = POMDPUtil.policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0, model=model, X=X, N=100, B_n=B_n)
    print(avg_cost)
    print(end-start)
