import time
from large_pomdp_parser import load_model
from rocksample_feature_parser import build_coarsened_model
from pomdp_util import POMDPUtil
from vi_2 import vi

if __name__ == '__main__':
    orig_model = load_model("rocksample_7_8.pkl")
    print("loaded model")
    n_grid=7
    x_res = 2
    y_res = 2
    model = build_coarsened_model(orig_model=orig_model, n=n_grid, x_res=x_res, y_res=y_res)
    print("built coarsened model")
    n = 1
    gamma = 0.95
    epsilon = 0.001
    X = list(model["state_index"].values())
    U = list(model["action_index"].values())
    O = list(model["obs_index"].values())
    b0 = list(model["start"])
    B_n = POMDPUtil.B_n(n=n, X=X)
    annoy_index = None
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
    end = time.time()
    print(end - start)
    X_orig = list(orig_model["state_index"].values())
    b0_orig = list(orig_model["start"])
    # avg_cost = POMDPUtil.policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0,
    #                                                model=model,
    #                                                X=X, N=100, B_n=B_n,
    #                                                annoy_index=annoy_index, b_to_index=b_to_index, n=n)
    avg_cost = POMDPUtil.feature_policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0_orig,
                                                   model=orig_model,
                                                   X=X_orig, N=100, B_n=B_n,n=n,
                                                   annoy_index=annoy_index, b_to_index=b_to_index,
                                                   feature_model=model, n_grid=n_grid, x_res=x_res, y_res=y_res)
    print(avg_cost)