import numpy as np
from pomdp_util import POMDPUtil

def compute_cost_of_belief(b, u, model):
    """
    Compute the cost of taking action u in belief b.
    For example, if your cost is the expected negative reward:
        cost(b,u) = sum_{x} b[x] * cost(u, x)
    or cost(u, x) = -R(u,x).

    Here, we show a simple version that uses model["R"]
    in the typical 'expected immediate cost' style:
        cost(b,u) = sum_{x} b[x] * [ sum_{x'} sum_{z} T[x->x']*Z[x',z]*( -R(u,x,x',z) ) ]
    or something simpler.  In many POMDPs, immediate cost depends
    only on (b,u), or on (x,u).  We'll do a simpler approach:

        cost(b,u) = sum_{x} b[x]* c(x,u)

    where c(x,u) we read from the model. We'll define an example
    c(x,u) = - expected reward from that state, or we look up
    a pre-stored cost_xu[u][x], etc.

    For demonstration, we do a stub that uses:
      cost_xu[u][x] = sum_{(x_next,z)} T[x->x_next]*Z[x_next,z]*(-R(u,x,x_next,z)).
    But we compute that on the fly or from a small cache.
    """

    # Suppose we do "cost(u,x) = - expected reward if it is stored".
    # We'll define a helper to compute cost(u,x) once and store it in a cache
    # to avoid repeated overhead, but if you want fully on-the-fly each time,
    # you can do so.

    # We'll assume we have a global or static cache: model["cost_xu"][u_id][x_id],
    # or we define a function:
    return on_the_fly_cost(b, u, model)


def on_the_fly_cost(b, u, model):
    """
    Example: cost(b,u) = sum_{x} b[x]* cost_xu(x,u).
    We'll define cost_xu(x,u) = sum_{(x',z)} T[a][x][x'] * Z[a][x'][z] * (-R(u,x,x',z)).

    For demonstration, here's a direct approach:
    """
    X = range(len(b))  # states
    a_id = u  # if u is numeric
    cost_val = 0.0
    for x_id, px in enumerate(b):
        if px == 0:
            continue
        # sum over next-states, observations
        succ = model["T"][a_id].get_successor_states(x_id)
        # get_successor_states => list of (x_next, pT)
        for (x_next, pT) in succ:
            if pT == 0.0:
                continue
            # Nonzero obs:
            obs_list = model["Z"][a_id].get_nonzero_observations(x_next)
            for (z_id, pO) in obs_list:
                # reward from model["R"] if present
                r = model["R"].get((a_id, x_id, x_next, z_id), 0.0)
                cost_val += px * pT * pO * (-r)  # cost = - reward
    return cost_val


def belief_update(b, a_id, z_id, model):
    """
    Standard POMDP belief update:
      b'(x') ~ sum_{x} b(x)*T[a][x][x'] * Z[a][x'][z], then normalized.
    Return b' or None if zero-likelihood.

    We return a *tuple* so it's hashable if we store in sets/dicts.
    """
    X = range(len(b))
    b_prime_unnorm = np.zeros(len(b), dtype=float)
    for x_id, px in enumerate(b):
        if px == 0.0:
            continue
        # transitions from x_id
        succ = model["T"][a_id].get_successor_states(x_id)
        for (x_next, pT) in succ:
            if pT == 0.0:
                continue
            pO = model["Z"][a_id].get_prob(x_next, z_id)
            if pO == 0.0:
                continue
            b_prime_unnorm[x_next] += px * pT * pO
    norm = b_prime_unnorm.sum()
    if norm == 0.0:
        return None
    b_prime = b_prime_unnorm / norm
    return tuple(b_prime)


def compute_next_belief_distribution(b, u, model, B_list, b_to_index, annoy_index):
    """
    Return list of (b_next_index, probability) for the 'belief MDP' transition,
    i.e. we consider all observations z => next belief b'. Probability is
    P(z|b,u).

    Because in belief MDP, from (b,u), the next 'state' is uncertain
    because we don't know which observation z will occur.
    We define p(z) = sum_{x} b[x]* sum_{x'} T[x->x']*Z[x'][z].

    Then next belief is b'(z). We'll find the index in B_list if it exists,
    or we skip if it's not in B_list (or you might do nearest neighbor, etc.).

    'B_list' is the enumerated set of beliefs. b_to_index is a dict that
    maps belief (as a tuple) to an integer index.
    """
    num_obs = len(model["observations"])
    transition_list = []
    for z_id in range(num_obs):
        # next belief:
        b_next = belief_update(b, u, z_id, model)
        if b_next is None:
            continue
        # Probability of that observation:
        # p(z|b,u) = sum_{x} b[x]* sum_{x'} T[x->x'] * Z[x'][z]
        # But we already effectively computed that in the normalization
        # if we look carefully.
        # For demonstration, let's compute pZ quickly:
        pZ = 0.0
        for x_id, px in enumerate(b):
            if px == 0.0:
                continue
            succ = model["T"][u].get_successor_states(x_id)
            for (x_next, pT) in succ:
                pZ += px * pT * model["Z"][u].get_prob(x_next, z_id)
        if pZ == 0.0:
            continue

        if b_next in b_to_index:
            j = b_to_index[b_next]
            transition_list.append((j, pZ))
        else:
            j = POMDPUtil.nearest_neighbor_annoy(b=b_next, annoy_index=annoy_index)
            transition_list.append((j, pZ))

        # else we skip or handle "nearest neighbor."
        # or you might add new beliefs dynamically.

    return transition_list


def vi(B_list, U, model, gamma, annoy_index, epsilon=1e-5, verbose=False):
    """
    Value iteration in the 'belief MDP' but computing transitions and cost
    on the fly from 'model'.

    B_list: list of beliefs (each a tuple or list).
    U: list of action indices, e.g. range(num_actions).
    model: parsed POMDP model with T, Z, R, etc.
    gamma: discount factor
    epsilon: stopping threshold
    Returns:
       policy 'mu' (array shape [len(B_list), len(U)])
         with a 1-hot for the chosen action at each belief,
       cost-to-go 'J' (array of shape [len(B_list)])
    """

    # We'll build a dict from beliefs to index for quick lookup
    b_to_index = {}
    for i, b in enumerate(B_list):
        # ensure it's a tuple
        if not isinstance(b, tuple):
            b = tuple(b)
            B_list[i] = b
        b_to_index[b] = i

    # Initialize the cost-to-go
    J = np.zeros(len(B_list), dtype=float)
    iteration = 0

    while True:
        delta = 0.0
        iteration += 1
        for i, b in enumerate(B_list):
            # compute Q_x[u] for each action
            Q_values = np.zeros(len(U), dtype=float)
            for a_id in U:
                # cost(b,a)
                immediate_cost = on_the_fly_cost(b, a_id, model)

                # next beliefs
                successors = compute_next_belief_distribution(
                    b, a_id, model, B_list, b_to_index, annoy_index)  # => list of (j, p)
                # sum_{j} p*( immediate_cost + gamma*J[j] )
                # But immediate_cost is added once, so Q_x[u] = immediate_cost + gamma * sum_{j} p* J[j]
                sum_future = 0.0
                for (j, p) in successors:
                    sum_future += p * J[j]

                Q_values[a_id] = immediate_cost + gamma * sum_future

            # pick best action
            best_a = np.argmin(Q_values)
            new_val = Q_values[best_a]
            # difference
            diff = abs(new_val - J[i])
            if diff > delta:
                delta = diff
            J[i] = new_val

        if verbose:
            print(f"Iteration {iteration}, delta={delta:.6g}, epsilon={epsilon}")
        if delta < epsilon:
            break

    # compute the final policy
    mu = np.zeros((len(B_list), len(U)), dtype=float)
    for i, b in enumerate(B_list):
        # compute best action
        Q_values = np.zeros(len(U), dtype=float)
        for a_id in U:
            immediate_cost = on_the_fly_cost(b, a_id, model)
            successors = compute_next_belief_distribution(b, a_id, model, B_list, b_to_index, annoy_index=annoy_index)
            sum_future = 0.0
            for (j, p) in successors:
                sum_future += p * J[j]
            Q_values[a_id] = immediate_cost + gamma * sum_future
        best_a = np.argmin(Q_values)
        mu[i, best_a] = 1.0

    return mu, J