import random
import math
import numpy as np
import itertools
from collections import deque
from large_pomdp_parser import get_next_state_obs_pairs, get_successor_states, sample_next_state_and_obs
from collections import defaultdict
from scipy.stats import qmc


class POMDPUtil:
    """
    Utility functions for belief aggregation in POMDPs
    """

    import numpy as np

    @staticmethod
    def belief_operator(z, u, b, X, model):
        """
        Computes the next belief b' after observing z in POMDP,
        given the prior belief b and action u.

        Parameters
        ----------
        z : int
            The observation index just received.
        u : int
            The action index taken.
        b : list or np.array
            The current belief distribution over states (length=len(X)).
        X : list or range of states (used for indexing).
        model : dict
            The parsed POMDP model, which includes:
              - model["Z"][u].get_prob(x_prime, z)
              - get_successor_states(model, x, u)

        Returns
        -------
        b_prime : np.array
            The updated normalized belief distribution (length=len(X)).
        """

        # b_prime accumulates unnormalized probabilities for each x'
        b_prime = np.zeros(len(X), dtype=float)
        norm = 0.0

        # Iterate only over states x with b[x] > 0
        # This can be a big speed-up if b is sparse
        nonzero_states = [x for x, prob_x in enumerate(b) if prob_x > 0.0]

        for x in nonzero_states:
            b_x = b[x]
            # Get successors (x_prime, t_p) = next state + transition prob
            transitions = get_successor_states(model, x, u)
            for (x_prime, t_p) in transitions:
                if t_p <= 0.0:
                    continue
                # Probability of observation z from x_prime under action u
                p_obs = model["Z"][u].get_prob(x_prime, z)
                if p_obs <= 0.0:
                    continue

                # Contribution to next-belief mass for x_prime
                contrib = b_x * t_p * p_obs
                b_prime[x_prime] += contrib
                norm += contrib

        # If norm == 0, we have an impossible observation under (b,u)
        if norm == 0.0:
            raise ValueError(f"Observation z={z} has zero probability under (b,u).")

        # Normalize
        b_prime /= norm

        # Check numerically that sum of b_prime is ~1
        if not np.allclose(b_prime.sum(), 1.0, rtol=1e-6, atol=1e-9):
            raise ValueError(
                f"Resulting belief does not sum to 1 (sum={b_prime.sum():.6f})."
            )

        return b_prime

    @staticmethod
    def nearest_neighbor(B_n, b):
        """
        Returns the nearest neighbor of b in B_n
        """
        distances = np.linalg.norm(np.array(B_n) - np.array(b), axis=1)
        nearest_index = int(np.argmin(distances))
        return B_n[nearest_index], nearest_index

    @staticmethod
    def B_n(n, X):
        """
        Creates the aggregate belief space B_n, where n is the resolution.
        Optimized to generate only valid combinations directly.
        """

        def generate_compositions(n, k):
            """Generates compositions of n into k parts (weak compositions)."""
            for indices in itertools.combinations_with_replacement(range(k), n):
                composition = [0] * k
                for index in indices:
                    composition[index] += 1
                yield composition

        belief_points = []
        total_combinations = math.comb(n + len(X) - 1, len(X) - 1)  # Number of compositions
        for i, k in enumerate(generate_compositions(n, len(X))):
            belief = [k_i / n for k_i in k]
            belief_points.append(belief)
            print(f"Processing belief {i + 1}/{total_combinations}")
        return belief_points

    @staticmethod
    def b_0_n(b0, B_n):
        """
        Creates the initial  belief
        """
        return POMDPUtil.nearest_neighbor(B_n=B_n, b=b0)[0]

    @staticmethod
    def C_b(B_n, X, U, model):
        """
        Compute the expected cost for each (belief, action) pair,
        using a precomputation of per-state-action costs.

        B_n: list of beliefs, each is something like a list of floats b[x] summing to 1
        X:   list/range of states
        U:   list/range of actions
        model: includes model["R"] reward dictionary, and get_next_state_obs_pairs

        Returns a 2D structure belief_C with shape [len(B_n)][len(U)]
        such that belief_C[i][u] = E[cost(b_i,u)].
        """
        # 1) Precompute cost_xu[u][x] = sum_{(xprime,z,p)} p * cost(u,x,xprime,z).
        #    cost(u,x,xprime,z) = -R(u,x,xprime,z).  If not in model["R"], it's 0.
        # We'll store in a 2D array or list of lists: cost_xu[u][x].

        num_actions = len(U)
        num_states = len(X)
        # We can store cost_xu as a 2D np array for speed
        cost_xu = np.zeros((num_actions, num_states), dtype=float)

        # Precompute for each (u, x):
        for a_id, a in enumerate(U):
            print(f"Precomputing cost {a_id}/{len(U)-1}")
            for x_id in X:
                # We'll sum up p * (-reward)
                total_cost = 0.0
                # get_next_state_obs_pairs => (x_prime, z, p_xz)
                nx_pairs = get_next_state_obs_pairs(model, x_id, a)
                for (x_prime, z, p_xz) in nx_pairs:
                    # Reward from model["R"]. If missing => 0
                    r = model["R"].get((a, x_id, x_prime, z), 0.0)
                    cost_value = -r  # cost = -reward
                    total_cost += p_xz * cost_value
                cost_xu[a_id, x_id] = total_cost

        # 2) Now for each belief b_i, cost(b_i, a_id) = sum_{x in X} b_i[x] * cost_xu[a_id,x].
        belief_C = np.zeros((len(B_n), num_actions), dtype=float)

        for i, b in enumerate(B_n):
            print(f"Processing belief {i}/{len(B_n) - 1}")
            # b is presumably a list or array of length len(X)
            # We'll do for each action
            #   cost = sum_{x} b[x] * cost_xu[a_id, x]
            for a_id, a in enumerate(U):
                # Dot product b . cost_xu[a_id]
                # If b is a list, convert or do a python sum
                # We'll do python sum for clarity:
                c = 0.0
                for x_id in X:
                    if b[x_id] != 0:
                        c += b[x_id] * cost_xu[a_id, x_id]
                belief_C[i, a_id] = c

        # Convert to lists if you prefer
        return belief_C.tolist()

    @staticmethod
    def expected_cost(b, u, z, C, X, P):
        """
        Computes E[C[x][u] | b]
        """
        # R[a][s][s'][o]
        return sum([C[u][x][x_prime][z] * b[x] * P[u][x][x_prime] for x in X for x_prime in X])

    @staticmethod
    def P_z_b_u(b, z, Z, X, U, P, u):
        """
        Computes P(z | b, u)
        """
        return sum([Z[x_prime][z] * b[x] * P[u][x][x_prime] for x in X for x_prime in X])

    @staticmethod
    def P_b(B_n, X, U, model):
        """
        Build a structure belief_succ[a][i] = a list of (j, prob),
        where j is the index of the next belief in B_n reached from b_i
        by action a, with probability prob = P(b_j | b_i, a).

        We'll store it in model["belief_succ"] so that we can quickly
        retrieve the nonzero next beliefs for any (b_i, a).

        This code uses:
          - get_next_state_obs_pairs(model, x, a) to get all (x_next, z, p)
            transitions from state x under action a.
          - POMDPUtil.belief_operator(z=z, u=a, b=b_i, ...) to compute the new belief b_prime
          - POMDPUtil.nearest_neighbor(...) to find which discrete belief in B_n is closest.

        We assume B_n[i] is the ith belief in the discrete set.
        """

        # Create a structure of shape [num_actions][num_beliefs]
        # each entry will eventually be a list of (next_belief_index, prob)
        num_beliefs = len(B_n)
        num_actions = len(U)
        B_succ = [[[] for _ in range(num_beliefs)] for _ in range(num_actions)]

        for a_id, a in enumerate(U):
            for i, b1 in enumerate(B_n):
                print(f"Processing action {a_id}/{len(U) - 1}, belief {i}/{len(B_n) - 1}")
                # We'll accumulate probabilities in a dictionary { j : prob }
                # where j is an index of B_n
                next_belief_prob = defaultdict(float)

                # for each state x with b1[x] > 0
                for x, weight_x in enumerate(b1):
                    if weight_x == 0:
                        continue
                    # get next state-obs pairs => (x_prime, z, p_xz)
                    nx_pairs = get_next_state_obs_pairs(model, x, a_id)
                    for (x_prime, z, p_xz) in nx_pairs:
                        if p_xz == 0:
                            continue
                        # compute the updated belief after observing z
                        b_prime = POMDPUtil.belief_operator(z=z, u=a_id, b=b1, X=X, model=model)
                        # find nearest neighbor in B_n
                        # (assuming nearest_neighbor returns an *index*, not an object)
                        j_idx = POMDPUtil.nearest_neighbor(B_n, b_prime)[1]

                        # accumulate
                        next_belief_prob[j_idx] += weight_x * p_xz

                # convert next_belief_prob dict into a list of (j, prob>0)
                # ignoring zeros
                for j_idx, pval in next_belief_prob.items():
                    if pval > 0:
                        B_succ[a_id][i].append((j_idx, pval))

        # store in the model
        model["belief_succ"] = B_succ
        return B_succ

    @staticmethod
    def particle_filter(particles, max_num_particles, P, o, u, X, O, Z):
        """
        Implements a particle filter
        """
        new_particles = []
        while len(new_particles) < max_num_particles:
            x = random.choice(particles)
            x_prime = np.random.choice(X, p=P[u][x])
            o_hat = np.random.choice(O, p=Z[u][x_prime])
            if o == o_hat:
                new_particles.append(x_prime)
        return new_particles

    @staticmethod
    def policy_evaluation(L, mu, gamma, b0, model, X, N, B_n):
        """
        Monte-Carlo evaluation of a policy
        """
        costs = []
        avg_cost = 0.0
        # B_n_prime = set([tuple(b) for b in B_n])
        for l in range(L):
            print(f"Policy evaluation iteration: {l}/{L}, avg cost: {avg_cost}")
            x = np.random.choice(X, p=b0)
            b = b0
            Cost = 0
            for k in range(N):
                # if tuple(b) not in B_n_prime:
                #     B_n_prime.add(tuple(b))
                b_idx = POMDPUtil.nearest_neighbor(B_n=B_n, b=b)[1]
                u = np.argmax(mu[b_idx])
                x_prime, z = sample_next_state_and_obs(model, x, u)
                r = 0
                if (u, x, x_prime, z) in model["R"]:
                    r = model["R"][(u, x, x_prime, z)]
                Cost += math.pow(gamma, k) * -r
                b = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, model=model)
                x = x_prime
            costs.append(Cost)
            avg_cost = np.mean(costs)
            # np.savez("B_n.npz", B_n=np.array([list(b) for b in B_n_prime]))
        return avg_cost

    @staticmethod
    def enumerate_reachable_beliefs(model, b0, U, X, T):
        """
        Enumerate all beliefs reachable from the initial belief b0
        (a list of length len(X)) within T steps, under *all possible* actions
        and observations.

        Parameters
        ----------
        model : dict
            The parsed POMDP model, containing "T", "Z", "R", etc.
            We assume we have a function 'belief_operator(z, u, b, X, model)'
            that returns b' = tau(b,u,z).
        b0 : list[float]
            The initial belief distribution over X.  sum(b0) = 1.
        X : list or range
            The state indices (0..S-1).
        U : list or range
            The action indices (0..A-1).
        T : int
            The maximum number of steps to expand forward.

        Returns
        -------
        reachable : list of (b, t)
            All (belief, time) pairs that are reachable.
            If you only want the beliefs (not time), you can store them in a set.

        Notes
        -----
        - The branching factor can be huge: #actions * possible #observations.
        - For T=100, this can blow up exponentially in many POMDPs.
          So in large domains, enumerating all reachable beliefs is usually infeasible.
        - You might consider a partial or approximate approach (see suggestions below).
        """
        # We'll store reachable beliefs in a BFS manner:
        # queue holds (b, t)
        queue = deque()
        # Use a set to store visited (b, t) or just b if you only care about
        # unique beliefs ignoring time dimension.
        visited = set()

        b0_t = tuple(b0)  # so it's hashable
        queue.append((b0_t, 0))
        visited.add((b0_t, 0))

        reachable = []  # or we can store them in a set
        reachable.append((b0_t, 0))

        while queue:
            (b_current, t) = queue.popleft()
            if t >= T:
                # we've reached the time horizon
                continue

            # For each action a
            for a_id in U:
                # For each possible observation z
                # We can figure out which obs are possible, but in worst case,
                # we iterate over all obs in model["observations"] => range(num_obs)
                num_obs = len(model["observations"])
                for z_id in range(num_obs):
                    try:
                        b_next = tuple(POMDPUtil.belief_operator(z=z_id, u=a_id, b=b_current, X=X, model=model))
                    except:
                        continue
                    # add (b_next, t+1) if not visited
                    if (b_next, t + 1) not in visited:
                        visited.add((b_next, t + 1))
                        queue.append((b_next, t + 1))
                        reachable.append((b_next, t + 1))
        return reachable

    @staticmethod
    def sample_beliefs_halton(num_beliefs, num_states):
        """
        Sample 'num_beliefs' beliefs in an S-dimensional simplex
        using a Halton sequence in dimension (S-1).
        """
        dim = num_states - 1
        # Construct a Halton sampler for dimension = S-1
        sampler = qmc.Halton(d=dim, scramble=False)
        points = sampler.random(n=num_beliefs)  # shape=(num_beliefs, dim)

        # Now convert each row in 'points' to a belief distribution
        # We can interpret them as "S-1 cut points" in [0,1].
        beliefs = []
        for row in points:
            sorted_cuts = np.sort(row)
            # e.g. if sorted_cuts=[0.2, 0.7] in the case of 3 states,
            # the intervals are [0,0.2],(0.2,0.7],(0.7,1], so belief=[0.2,0.5,0.3].
            cuts_full = np.concatenate(([0.0], sorted_cuts, [1.0]))
            diffs = np.diff(cuts_full)
            beliefs.append(diffs.tolist())

        return beliefs

    @staticmethod
    def sample_beliefs_dirichlet(num_beliefs, num_states, alpha=1.0):
        """
        Sample 'num_beliefs' random beliefs in an S-dimensional simplex,
        by sampling from a Dirichlet(alpha,...,alpha) distribution.

        alpha : float or list of length 'num_states'
            If alpha is a scalar, we use Dirichlet(alpha,...,alpha).
            If alpha is a list, we do Dirichlet(alpha_i,...).

        Returns
        -------
        beliefs : list of lists
            Each belief is a length-num_states list summing to 1.
        """
        # If alpha is a scalar, replicate it
        if isinstance(alpha, (int, float)):
            alpha_vec = np.full(num_states, alpha)
        else:
            alpha_vec = np.array(alpha)
            assert len(alpha_vec) == num_states

        beliefs = []
        for _ in range(num_beliefs):
            b = np.random.gamma(shape=alpha_vec, scale=1.0)
            b /= b.sum()  # normalize
            beliefs.append(b.tolist())
        return beliefs



