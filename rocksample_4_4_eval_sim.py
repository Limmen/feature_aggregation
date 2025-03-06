import sys

from sympy.abc import alpha

from vi_2 import vi
from pomdp_util import POMDPUtil
import time
from large_pomdp_parser import load_model
from rocksample_simulator import RockSampleSimulator
from q_learning import run_q_learning_in_Bn

if __name__ == '__main__':
    env = RockSampleSimulator(n=4, k=4, seed=999)
    gamma = 0.95
    n = 2
    epsilon = 0.001
    X = list(range(env.num_states))
    U = list(range(env.num_actions))
    O = list(range(env.num_observations))
    b0 = env.initial_belief()
    B_n = POMDPUtil.B_n(n=n, X=X)
    b_0_n = POMDPUtil.b_0_n(b0=b0, B_n=B_n)
    B_n_indices = [i for i in range(len(B_n))]
    b_0_n_idx = B_n.index(b_0_n)
    run_q_learning_in_Bn(env=env, B_n=B_n, gamma=gamma, alpha=0.3, epsilon=0.5, num_iterations=100000, log_frequency=100,
                         start_belief_id=b_0_n_idx, n=n)

    # start = time.time()
    # mu, J = vi(B_n=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, cost_xu=cost_xu,
    #            annoy_index=annoy_index, b_to_index=b_to_index, n=n)
    # # mu, J = vi_3(B_list=B_n, U=U, gamma=gamma, epsilon=epsilon, verbose=True, model=model, annoy_index=annoy_index)
    # end = time.time()
    # print(end-start)
    # avg_cost = POMDPUtil.policy_evaluation_simulator(L=1000, mu=mu, gamma=gamma, model=model, X=X, N=100,
    #                                                  b_to_index=b_to_index, n=n)
    # print(avg_cost)
