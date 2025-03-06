from pomdp_util import POMDPUtil
from rocksample_simulator import RockSampleSimulator
from vi_5 import vi
import time

if __name__ == '__main__':
    env = RockSampleSimulator(n=10, k=10, seed=999)
    gamma = 0.95
    n = 1
    epsilon = 0.001
    X = list(range(env.num_states))
    U = list(range(env.num_actions))
    O = list(range(env.num_observations))
    b0 = env.initial_belief()
    B_n = POMDPUtil.B_n(n=n, X=X)
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    b_to_index = {}
    print("Creating b_to_index")
    for i, b in enumerate(B_n):
        if not isinstance(b, tuple):
            b = tuple(b)
            B_n[i] = b
        b_to_index[b] = i
    print("Creating cost_xu")
    cost_xu = env.precompute_cost()
    # print(cost_xu[6])
    # import sys
    # sys.exit(0)
    start = time.time()
    mu, J = vi(B_n=B_n, U=U, env=env, gamma=gamma, epsilon=epsilon, verbose=True, n=n, cost_xu=cost_xu,
               b_to_index=b_to_index)
    end = time.time()
    # print(end - start)
    # avg_cost = env.policy_evaluation(L=1000, mu=mu, b0=b0, N=100, b_to_index=b_to_index, n=n, gamma=gamma)
    # avg_cost = POMDPUtil.policy_evaluation(L=1000, mu=mu, gamma=gamma, b0=b0, model=model, X=X, N=100, B_n=B_n,
    #                                        annoy_index=None, b_to_index=b_to_index, n=n)
    # print(avg_cost)

    # run_q_learning_in_Bn(env=env, B_n=B_n, gamma=gamma, alpha=0.3, epsilon=0.5, num_iterations=100000, log_frequency=100,
    #                      start_belief_id=b_0_n_idx, n=n)

    # start = time.time()
    # mu, J = vi(B_n=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, cost_xu=cost_xu,
    #            annoy_index=annoy_index, b_to_index=b_to_index, n=n)
    # # mu, J = vi_3(B_list=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, annoy_index=annoy_index)
    # end = time.time()
    # print(end-start)
    # avg_cost = POMDPUtil.policy_evaluation_simulator(L=1000, mu=mu, gamma=gamma, model=model, X=X, N=100,
    #                                                  b_to_index=b_to_index, n=n)
    # print(avg_cost)
