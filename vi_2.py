import numpy as np
from large_pomdp_parser import get_next_state_obs_pairs, get_successor_states
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


def compute_next_belief_distribution(b, u, model, annoy_index, B_n, b_to_index, n):
    """
    Return list of (b_next_index, probability) for the 'belief MDP' transition,
    """
    transition_dict = {}
    for x_id, px in enumerate(b):
        if px == 0.0:
            continue
        nx_pairs = get_next_state_obs_pairs(model, x_id, u)
        for (x_prime, z, p_xz) in nx_pairs:
            # b_next = POMDPUtil.nearest_neighbor_annoy(
            #     b=POMDPUtil.belief_operator(z=z, u=u, b=b, X=model["states"], model=model), annoy_index=annoy_index,
            #     b_to_index=b_to_index)
            # if b_next in transition_dict:
            #     transition_dict[b_next] += px * p_xz
            # else:
            #     transition_dict[b_next] = px * p_xz
            # b_next = POMDPUtil.nearest_neighbor(
            #     b=POMDPUtil.belief_operator(z=z, u=u, b=b, X=model["states"], model=model), B_n=B_n)[1]
            b_next = POMDPUtil.belief_operator(z=z, u=u, b=b, X=model["states"], model=model)
            if tuple(b_next) in b_to_index:
                j = b_to_index[tuple(b_next)]
            else:
                j = b_to_index[tuple(POMDPUtil.round_to_Bn(b=b, n=n))]
            if j in transition_dict:
                transition_dict[j] += px * p_xz
            else:
                transition_dict[j] = px * p_xz
    return transition_dict


def vi(B_n, U, model, gamma, cost_xu, annoy_index, b_to_index, n, epsilon=1e-5, verbose=False):
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
            successors_dict[i][u] = compute_next_belief_distribution(b=b, u=u, model=model, annoy_index=annoy_index,
                                                                     B_n=B_n, b_to_index=b_to_index, n=n)
            # successors = compute_next_belief_distribution(b=b, u=u, model=model, annoy_index=annoy_index)

    while True:
        delta = 0.0
        iteration += 1
        for i, b in enumerate(B_n):
            # if verbose and i % 1 == 0:
            #     print(f"VI iteration: {iteration}, belief coverage: {(int((i / len(B_n))* 100)) } %")
            Q_values = np.zeros(len(U), dtype=float)
            for u in U:
                immediate_cost = expected_cost(b, u, cost_xu)
                successors = successors_dict[i][u]
                # successors = compute_next_belief_distribution(b=b, u=u, model=model, annoy_index=annoy_index)
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
            print(f"Iteration {iteration}, delta={delta:.6g}, epsilon={epsilon}")
        if delta < epsilon:
            break

    # compute the final policy
    mu = np.zeros((len(B_n), len(U)), dtype=float)
    for i, b in enumerate(B_n):
        Q_values = np.zeros(len(U), dtype=float)
        for u in U:
            immediate_cost = expected_cost(b, u, cost_xu)
            successors = successors_dict[i][u]
            # successors = compute_next_belief_distribution(b=b, u=u, model=model, annoy_index=annoy_index)
            sum_future = 0.0
            for j, p in successors.items():
                sum_future += p * J[j]
            Q_values[u] = immediate_cost + gamma * sum_future
        best_a = np.argmin(Q_values)
        mu[i, best_a] = 1.0

    return mu, J
