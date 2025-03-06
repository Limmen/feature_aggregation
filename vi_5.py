import numpy as np
from rocksample_simulator import RockSampleSimulator
from pomdp_util import POMDPUtil


def expected_cost(b, u, cost_xu):
    """
    Calculates g(b,u)
    """
    cost_val = 0.0
    for x_id, px in enumerate(b):
        if px == 0:
            continue
        cost_val += b[x_id] * cost_xu[u, x_id]
    return cost_val


def compute_next_belief_distribution(b, u, env: RockSampleSimulator, b_to_index, n):
    """
    Return list of (b_next_index, probability) for the 'belief MDP' transition,
    """
    transition_dict = {}
    for x_id, px in enumerate(b):
        if px == 0.0:
            continue
        nx_pairs = env.next_state_obs_distribution(x_id, u)
        for (x_prime, z, p_xz) in nx_pairs:
            b_next = env.belief_operator_pf(z_id=z, a_id=u, b=b, num_particles=50)
            if tuple(b_next) in b_to_index:
                j = b_to_index[tuple(b_next)]
            else:
                j = b_to_index[tuple(POMDPUtil.round_to_Bn(b=b, n=n))]
            if j in transition_dict:
                transition_dict[j] += px * p_xz
            else:
                transition_dict[j] = px * p_xz
    return transition_dict


def vi(B_n, U, env, gamma, cost_xu, b_to_index, n, epsilon=1e-5, verbose=False):
    """
    Simulation-based value iteration in belief space that exploits sparsity of the POMDP.
    """
    J = np.zeros(len(B_n), dtype=float)
    iteration = 0

    successors_dict = {}
    for i, b in enumerate(B_n):
        successors_dict[i] = {}
        for u in U:
            print(f"{i}/{len(B_n)}, {u}/{len(U)}")
            successors_dict[i][u] = compute_next_belief_distribution(b=b, u=u, env=env, b_to_index=b_to_index, n=n)

    while True:
        delta = 0.0
        iteration += 1
        for i, b in enumerate(B_n):
            Q_values = np.zeros(len(U), dtype=float)
            for u in U:
                immediate_cost = expected_cost(b, u, cost_xu)
                successors = successors_dict[i][u]
                sum_future = 0.0
                for j, p in successors.items():
                    sum_future += p * J[j]
                Q_values[u] = immediate_cost + gamma * sum_future

            best_a = np.argmin(Q_values)
            new_val = Q_values[best_a]
            diff = abs(new_val - J[i])
            if diff > delta:
                delta = diff
            J[i] = new_val
        if verbose:
            mu = create_policy(B_n, U, successors_dict, cost_xu, J, gamma)
            avg_cost = env.policy_evaluation(L=100, mu=mu, b0=env.initial_belief(), N=100,
                                             b_to_index=b_to_index, n=n, gamma=gamma, use_pf=True)
            print(f"Iteration {iteration}, delta={delta:.6g}, epsilon={epsilon}, cost: {avg_cost}")
        if delta < epsilon:
            break

    # compute the final policy
    mu = create_policy(B_n, U, successors_dict, cost_xu, J, gamma)
    return mu, J


def create_policy(B_n, U, successors_dict, cost_xu, J, gamma):
    mu = np.zeros((len(B_n), len(U)), dtype=float)
    for i, b in enumerate(B_n):
        Q_values = np.zeros(len(U), dtype=float)
        for u in U:
            immediate_cost = expected_cost(b, u, cost_xu)
            successors = successors_dict[i][u]
            sum_future = 0.0
            for j, p in successors.items():
                sum_future += p * J[j]
            Q_values[u] = immediate_cost + gamma * sum_future
        best_a = np.argmin(Q_values)
        mu[i, best_a] = 1.0
    return mu
