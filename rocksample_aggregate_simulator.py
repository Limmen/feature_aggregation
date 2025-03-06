#!/usr/bin/env python3

import random
import numpy as np
import math
from pomdp_util import POMDPUtil


###############################################################################
# ROCKSAMPLE SIMULATOR (Original)
###############################################################################
class RockSampleSimulator:
    """
    Integer-encoded RockSample(n, k) simulator.

    STATES:
      - Each state is an integer in [0, n*n*2^k - 1].
      - decode_state(...) => (x, y, bitmask)
      - encode_state(x, y, bitmask) => integer

    ACTIONS (integer):
      0 -> 'north'
      1 -> 'south'
      2 -> 'east'
      3 -> 'west'
      4 -> 'sample'
      5+i -> 'sense_i'  for i in [0..k-1]

    OBSERVATIONS (integer):
      0 -> 'none'
      1 -> 'good'
      2 -> 'bad'

    REWARDS:
      step_cost, sample_good_reward, sample_bad_reward, exit_reward, etc.

    The environment also stores fixed rock positions in self.rock_positions[i] = (rx, ry).
    """

    def __init__(self, n, k,
                 step_cost=0,
                 sample_good_reward=10,
                 sample_bad_reward=-10,
                 exit_reward=10,
                 seed=None):
        """
        :param n: grid dimension
        :param k: number of rocks
        :param step_cost: cost per action (often 0 or negative)
        :param sample_good_reward: reward if sampling a good rock
        :param sample_bad_reward: reward if sampling a bad rock
        :param exit_reward: reward if the agent steps beyond x==n
        :param seed: random seed for reproducibility
        """
        self.n = n
        self.k = k
        self.step_cost = step_cost
        self.sample_good_reward = sample_good_reward
        self.sample_bad_reward = sample_bad_reward
        self.exit_reward = exit_reward

        if seed is not None:
            random.seed(seed)

        # Assign rock positions (distinct random positions).
        self.rock_positions = []
        used = set()
        while len(self.rock_positions) < k:
            rx = random.randint(0, n - 1)
            ry = random.randint(0, n - 1)
            if (rx, ry) not in used:
                used.add((rx, ry))
                self.rock_positions.append((rx, ry))

        # 1) Build state <-> ID mappings
        self.num_states = n * n * (2 ** k)
        self.state_id2info = [None] * self.num_states
        self.info2state_id = {}

        idx = 0
        for x in range(n):
            for y in range(n):
                for bitmask in range(2 ** k):
                    self.state_id2info[idx] = (x, y, bitmask)
                    self.info2state_id[(x, y, bitmask)] = idx
                    idx += 1

        # 2) Build action <-> ID
        self.num_actions = 5 + k
        self.action_id2str = {}
        self.action_str2id = {}

        base_actions = ["north", "south", "east", "west", "sample"]
        for i, a_str in enumerate(base_actions):
            self.action_id2str[i] = a_str
            self.action_str2id[a_str] = i

        for i in range(k):
            sense_str = f"sense_{i}"
            sense_id = 5 + i
            self.action_id2str[sense_id] = sense_str
            self.action_str2id[sense_str] = sense_id

        # 3) Observations
        self.num_observations = 3
        self.obs_id2str = {0: 'none', 1: 'good', 2: 'bad'}
        self.obs_str2id = {'none': 0, 'good': 1, 'bad': 2}

    # ----------------------------
    # State encoding / decoding
    # ----------------------------
    def encode_state(self, x, y, bitmask):
        return (x * self.n + y) * (2 ** self.k) + bitmask

    def decode_state(self, state_id):
        return self.state_id2info[state_id]

    # ----------------------------
    # Initial conditions
    # ----------------------------
    def initial_belief(self):
        """
        Returns a probability vector over all states,
        with (x=0,y=0) and any rock bitmask equally likely (1/(2^k)).
        """
        b = [0.0] * self.num_states
        p = 1.0 / (2 ** self.k)
        for bitmask in range(2 ** self.k):
            s_id = self.encode_state(0, 0, bitmask)
            b[s_id] = p
        return b

    def init_state(self):
        """
        Returns an initial *concrete* state with x=0, y=0, each rock good/bad w.p. 0.5
        """
        x, y = 0, 0
        bitmask = 0
        for i in range(self.k):
            if random.random() < 0.5:
                bitmask |= (1 << i)
        return self.encode_state(x, y, bitmask)

    # ----------------------------
    # Movement checks
    # ----------------------------
    def valid_position(self, x, y):
        return (0 <= x < self.n) and (0 <= y < self.n)

    # ----------------------------
    # Single-step environment logic
    # ----------------------------
    def step(self, state_id, action_id):
        """
        Perform a single step with the given action_id from state_id.
        Returns (next_state_id, obs_id, cost, done).

        NOTE: The returned 'cost' = - (internal reward).
        So if internally we compute reward = +10, the returned cost is -10.
        """
        x, y, bitmask = self.decode_state(state_id)
        reward = 0
        obs_id = self.obs_str2id['none']
        done = False

        if action_id not in self.action_id2str:
            raise ValueError(f"Invalid action_id={action_id}")
        action_str = self.action_id2str[action_id]

        # Base step cost
        reward += self.step_cost

        if action_str == 'north':
            nx, ny = x, y + 1
            if self.valid_position(nx, ny):
                x, y = nx, ny

        elif action_str == 'south':
            nx, ny = x, y - 1
            if self.valid_position(nx, ny):
                x, y = nx, ny

        elif action_str == 'east':
            nx, ny = x + 1, y
            if self.valid_position(nx, ny):
                x, y = nx, ny
            else:
                # stepping beyond x==n
                if nx == self.n:
                    reward += self.exit_reward
                    done = True

        elif action_str == 'west':
            nx, ny = x - 1, y
            if self.valid_position(nx, ny):
                x, y = nx, ny

        elif action_str == 'sample':
            # Check if the agent is on a rock
            rock_found = False
            for i, (rx, ry) in enumerate(self.rock_positions):
                if rx == x and ry == y:
                    rock_found = True
                    # Check if bit i is set => good
                    if (bitmask & (1 << i)) != 0:
                        reward += self.sample_good_reward
                    else:
                        reward += self.sample_bad_reward
                    # Clear that bit => rock is now worthless
                    bitmask &= ~(1 << i)
                    break
            # If we did not find any rock, impose penalty
            if not rock_found:
                reward -= 100

        elif action_str.startswith('sense_'):
            i_rock = int(action_str.split('_')[1])
            rx, ry = self.rock_positions[i_rock]
            dist = abs(x - rx) + abs(y - ry)
            correct_prob = 1.0 - 0.5 ** dist
            is_good = ((bitmask & (1 << i_rock)) != 0)

            # sample observation
            if random.random() < correct_prob:
                obs_id = self.obs_str2id['good'] if is_good else self.obs_str2id['bad']
            else:
                obs_id = self.obs_str2id['bad'] if is_good else self.obs_str2id['good']

        else:
            raise ValueError(f"Unknown action string: {action_str}")

        next_state_id = self.encode_state(x, y, bitmask)

        # If we stepped off the east edge exactly:
        if (not done) and (x == self.n):
            done = True

        # Return cost = -reward
        return next_state_id, obs_id, float(-reward), done

    # ----------------------------
    # Next-state/obs distribution
    # ----------------------------
    def next_state_obs_distribution(self, state_id, action_id):
        """
        Return a list of (next_state_id, obs_id, probability)
        enumerating all possible next states & observations
        from (state_id, action_id).
        """
        x, y, bitmask = self.decode_state(state_id)
        if action_id not in self.action_id2str:
            raise ValueError(f"Invalid action_id {action_id}")
        a_str = self.action_id2str[action_id]

        nx, ny, nbitmask = x, y, bitmask
        possible_obs = []

        # 1) Determine next state ignoring sense obs randomness
        if a_str == 'north':
            maybe_y = y + 1
            if self.valid_position(x, maybe_y):
                ny = maybe_y

        elif a_str == 'south':
            maybe_y = y - 1
            if self.valid_position(x, maybe_y):
                ny = maybe_y

        elif a_str == 'east':
            maybe_x = x + 1
            if self.valid_position(maybe_x, y):
                nx = maybe_x
            else:
                # could be stepping out
                pass

        elif a_str == 'west':
            maybe_x = x - 1
            if self.valid_position(maybe_x, y):
                nx = maybe_x

        elif a_str == 'sample':
            # If on a rock, clear bit for that rock
            for i, (rx, ry) in enumerate(self.rock_positions):
                if rx == x and ry == y:
                    nbitmask = nbitmask & ~(1 << i)
                    break
            # obs = none

        elif a_str.startswith('sense_'):
            # next state remains the same (except for potential boundary clamp)
            pass

        else:
            raise ValueError(f"Unknown action '{a_str}'")

        # clamp or check bounds
        if not (0 <= nx < self.n) or not (0 <= ny < self.n):
            nx = max(0, min(nx, self.n - 1))
            ny = max(0, min(ny, self.n - 1))

        next_s_id = self.encode_state(nx, ny, nbitmask)

        # 2) define obs distribution
        if a_str.startswith('sense_'):
            i_rock = int(a_str.split('_')[1])
            rx, ry = self.rock_positions[i_rock]
            dist = abs(nx - rx) + abs(ny - ry)
            correct_prob = 1.0 - 0.5 ** dist
            is_good = (nbitmask & (1 << i_rock)) != 0
            obs_good = self.obs_str2id['good']
            obs_bad = self.obs_str2id['bad']

            if is_good:
                possible_obs.append((obs_good, correct_prob))
                possible_obs.append((obs_bad, 1.0 - correct_prob))
            else:
                possible_obs.append((obs_good, 1.0 - correct_prob))
                possible_obs.append((obs_bad, correct_prob))
        else:
            # single obs = none
            obs_none = self.obs_str2id['none']
            possible_obs.append((obs_none, 1.0))

        # build final list
        result = []
        for (obs_id, p) in possible_obs:
            if p > 0.0:
                result.append((next_s_id, obs_id, p))

        return result

    # ----------------------------
    # Belief Update
    # ----------------------------
    def belief_operator(self, b, a_id, z_id):
        """
        Standard (exact) belief update given action a_id and observation z_id.
        b'(x') = (1 / norm) * sum_x [ b[x] * P(x'|x,a) * P(z|x',a) ]
        """
        num_states = self.num_states
        b_prime = np.zeros(num_states, dtype=float)

        for x_id, p_x in enumerate(b):
            if p_x == 0.0:
                continue
            nx_distribution = self.next_state_obs_distribution(x_id, a_id)
            for (nx_id, obs, p_trans) in nx_distribution:
                if obs == z_id:
                    b_prime[nx_id] += p_x * p_trans

        norm = b_prime.sum()
        if norm <= 1e-15:
            # If this happens, it's typically a "zero-likelihood" observation
            # We can throw an error or fallback to uniform.
            raise ValueError(f"Zero-likelihood observation {z_id} for belief update.")

        b_prime /= norm
        return b_prime

    def observation_probability(self, x_id, a_id, z_id):
        """
        Return P(z_id | x_id, a_id).
        """
        a_str = self.action_id2str[a_id]
        x, y, bitmask = self.decode_state(x_id)

        if a_str.startswith('sense_'):
            i_rock = int(a_str.split('_')[1])
            rx, ry = self.rock_positions[i_rock]
            dist = abs(x - rx) + abs(y - ry)
            correct_prob = 1.0 - (0.5 ** dist)
            rock_good = ((bitmask & (1 << i_rock)) != 0)

            if rock_good:
                if z_id == self.obs_str2id['good']:
                    return correct_prob
                elif z_id == self.obs_str2id['bad']:
                    return 1.0 - correct_prob
                else:
                    return 0.0
            else:
                if z_id == self.obs_str2id['bad']:
                    return correct_prob
                elif z_id == self.obs_str2id['good']:
                    return 1.0 - correct_prob
                else:
                    return 0.0
        else:
            # move / sample => always 'none'
            if z_id == self.obs_str2id['none']:
                return 1.0
            else:
                return 0.0

    def sample_next_state(self, x_id, a_id):
        """
        Sample a next state x' from P(x' | x_id, a_id).
        """
        dist_dict = {}
        for (nx_id, obs_id, p) in self.next_state_obs_distribution(x_id, a_id):
            dist_dict[nx_id] = dist_dict.get(nx_id, 0.0) + p

        nx_ids = list(dist_dict.keys())
        probs = np.array([dist_dict[nx] for nx in nx_ids], dtype=float)
        total_p = probs.sum()
        if total_p < 1e-15:
            # Degenerate fallback: stay in same state
            return x_id
        probs /= total_p

        chosen_idx = np.random.choice(len(nx_ids), p=probs)
        return nx_ids[chosen_idx]

    def belief_operator_pf(self, b, a_id, z_id, num_particles=500, eps=1e-12):
        """
        Particle filter update for belief distribution.
        """
        num_states = self.num_states
        state_ids = np.arange(num_states)
        # sample from current belief
        particles = np.random.choice(state_ids, size=num_particles, p=b)

        next_particles = np.zeros(num_particles, dtype=int)
        weights = np.zeros(num_particles, dtype=float)

        for i in range(num_particles):
            x_cur = particles[i]
            x_next = self.sample_next_state(x_cur, a_id)
            next_particles[i] = x_next
            w = self.observation_probability(x_next, a_id, z_id)
            weights[i] = w

        w_sum = weights.sum()
        if w_sum < eps:
            # fallback to uniform
            return np.ones(num_states) / num_states
        else:
            weights /= w_sum
            idx_resampled = np.random.choice(num_particles, size=num_particles, p=weights)
            resampled_particles = next_particles[idx_resampled]

            b_prime = np.zeros(num_states, dtype=float)
            for x_p in resampled_particles:
                b_prime[x_p] += 1.0
            b_prime /= num_particles
            return b_prime

    def get_reward(self, x_id, a_id, x_next_id, obs_id):
        """
        Return the environment's immediate REWARD for (x,a,x_next,obs).
        step(...) returns the negative of this as 'cost'.
        """
        x, y, bitmask = self.decode_state(x_id)
        nx, ny, nbitmask = self.decode_state(x_next_id)
        a_str = self.action_id2str[a_id]

        rew = 0.0
        # step cost
        rew += self.step_cost

        # check exit if action is east
        if a_str == 'east':
            # if next_state didn't move but we tried to step out => exit reward
            if (nx == x) and (x + 1 == self.n):
                rew += self.exit_reward

        # sample logic
        if a_str == 'sample':
            rock_found = False
            for i, (rx, ry) in enumerate(self.rock_positions):
                if rx == x and ry == y:
                    rock_found = True
                    if (bitmask & (1 << i)) != 0:
                        rew += self.sample_good_reward
                    else:
                        rew += self.sample_bad_reward
                    break
            if not rock_found:
                rew -= 100

        # sense has no direct reward other than the step cost
        return rew

    def precompute_cost(self):
        """
        Returns a 2D array cost_xu[a, s], where
           cost_xu[a,s] = E[ - reward ] = sum_{(s',obs)} p(s',obs|s,a)* ( - get_reward(...) ).
        """
        cost_xu = np.zeros((self.num_actions, self.num_states), dtype=float)

        for a_id in range(self.num_actions):
            for s_id in range(self.num_states):
                total_cost = 0.0
                nx_dist = self.next_state_obs_distribution(s_id, a_id)  # (ns, obs, p)
                for (ns, obs, p_nsobs) in nx_dist:
                    r = self.get_reward(s_id, a_id, ns, obs)
                    cost_val = -r
                    total_cost += p_nsobs * cost_val
                cost_xu[a_id, s_id] = total_cost

        return cost_xu

    def policy_evaluation(self, L, mu, gamma, b0, N,
                          B_n=None, b_to_index=None, n=4, use_pf=False):
        """
        Example function that simulates a policy mu for L episodes,
        each up to N steps, starting from belief b0.
        :param L: number of episodes to average over
        :param mu: a policy array of shape [|B|, num_actions],
                   or some function that returns an action from a belief index
        :param gamma: discount factor
        :param b0: initial belief distribution (numpy array of length self.num_states)
        :param N: max number of steps per episode
        :param B_n: (optional) a set of representative beliefs, if you do approximate indexing
        :param b_to_index: dictionary mapping belief -> index in mu
        :param n: rounding parameter if you do rounding in b
        :param use_pf: if True, do PF belief updates, else do exact.
        :return: average cost
        """
        costs = []
        avg_cost = 0.0

        for l in range(L):
            # Initialize from b0
            s_id = np.random.choice(self.num_states, p=b0)
            b = b0.copy()
            total_cost = 0.0

            for step_idx in range(N):
                # If we keep a dictionary b->index, we can look up the action:
                if b_to_index is not None:
                    # Attempt direct lookup, else round
                    if tuple(b) in b_to_index:
                        b_idx = b_to_index[tuple(b)]
                    else:
                        b_rounded = POMDPUtil.round_to_Bn(b, n=n)
                        b_idx = b_to_index[tuple(b_rounded)]
                    a_id = np.argmax(mu[b_idx])
                else:
                    # fallback: pick random or something
                    a_id = np.random.randint(self.num_actions)

                # environment step
                next_s_id, obs_id, cost, done = self.step(s_id, a_id)
                total_cost += (gamma ** step_idx) * cost

                # belief update
                if use_pf:
                    b = self.belief_operator_pf(b, a_id, obs_id, num_particles=100)
                else:
                    b = self.belief_operator(b, a_id, obs_id)

                s_id = next_s_id
                if done:
                    break

            costs.append(total_cost)
            avg_cost = np.mean(costs)
        return avg_cost


###############################################################################
# AGGREGATE ROCKSAMPLE SIMULATOR
###############################################################################
class AggregateRockSampleSimulator:
    """
    A wrapper around a RockSampleSimulator that aggregates the (x,y) grid
    into larger cells. The bitmask dimension remains un-aggregated.

    - The aggregator grid has dimension n_agg = ceil(n / grid_resolution).
    - State space size = n_agg * n_agg * 2^k.
    - We replicate many methods (step, next_state_obs_distribution, etc.)
      but at the aggregated level.
    """

    def __init__(self, base_simulator: RockSampleSimulator, grid_resolution=2):
        """
        :param base_simulator: an instance of RockSampleSimulator
        :param grid_resolution: how many original cells each aggregator cell spans
        """
        self.base_sim = base_simulator
        self.grid_resolution = grid_resolution

        self.n = base_simulator.n
        self.k = base_simulator.k

        # aggregator grid dimension
        self.n_agg = math.ceil(self.n / self.grid_resolution)

        # total aggregator states
        self.num_agg_states = self.n_agg * self.n_agg * (2 ** self.k)

        # same actions / observations as base
        self.num_actions = self.base_sim.num_actions
        self.action_id2str = self.base_sim.action_id2str
        self.action_str2id = self.base_sim.action_str2id

        self.num_observations = self.base_sim.num_observations
        self.obs_id2str = self.base_sim.obs_id2str
        self.obs_str2id = self.base_sim.obs_str2id

        self.rock_positions = self.base_sim.rock_positions  # original coords

        # aggregator ID <-> (xa, ya, bitmask)
        self.agg_id2info = [None] * self.num_agg_states
        self.agg_info2id = {}
        idx = 0
        for xa in range(self.n_agg):
            for ya in range(self.n_agg):
                for bm in range(2 ** self.k):
                    self.agg_id2info[idx] = (xa, ya, bm)
                    self.agg_info2id[(xa, ya, bm)] = idx
                    idx += 1

        # For aggregator cell <-> underlying (x,y)
        self.position_to_agg = {}
        for x in range(self.n):
            for y in range(self.n):
                xa = x // self.grid_resolution
                ya = y // self.grid_resolution
                self.position_to_agg[(x, y)] = (xa, ya)

        self.agg_to_positions = {}
        for xa in range(self.n_agg):
            for ya in range(self.n_agg):
                positions = []
                x_min = xa * self.grid_resolution
                x_max = min(self.n, (xa + 1) * self.grid_resolution)
                y_min = ya * self.grid_resolution
                y_max = min(self.n, (ya + 1) * self.grid_resolution)
                for xx in range(x_min, x_max):
                    for yy in range(y_min, y_max):
                        positions.append((xx, yy))
                self.agg_to_positions[(xa, ya)] = positions

    # ------------------------------------------------
    # Basic aggregator helpers
    # ------------------------------------------------
    def encode_agg_state(self, xa, ya, bitmask):
        return (xa * self.n_agg + ya) * (2 ** self.k) + bitmask

    def decode_agg_state(self, agg_state_id):
        return self.agg_id2info[agg_state_id]

    def aggregator_position(self, x, y):
        return self.position_to_agg[(x, y)]

    def aggregator_to_underlying_positions(self, xa, ya):
        return self.agg_to_positions[(xa, ya)]

    # ------------------------------------------------
    # next_state_obs_distribution (aggregate)
    # ------------------------------------------------
    def next_state_obs_distribution(self, agg_state_id, action_id):
        """
        Return a list of (next_agg_state_id, obs_id, probability),
        combining all underlying transitions from the original simulator.
        We assume uniform probability among the underlying states
        that map to this aggregator state.
        """
        xa, ya, bm = self.decode_agg_state(agg_state_id)
        positions = self.aggregator_to_underlying_positions(xa, ya)
        if len(positions) == 0:
            return []

        p_each = 1.0 / len(positions)
        dist = {}  # dist[next_agg_id][obs_id] = prob

        for (ux, uy) in positions:
            s_id = self.base_sim.encode_state(ux, uy, bm)
            ns_obs_list = self.base_sim.next_state_obs_distribution(s_id, action_id)
            for (ns_id, obs_id, p_nsobs) in ns_obs_list:
                p_contrib = p_each * p_nsobs
                if p_contrib == 0.0:
                    continue

                (nx, ny, nbm) = self.base_sim.decode_state(ns_id)
                xna, yna = self.aggregator_position(nx, ny)
                next_agg_id = self.encode_agg_state(xna, yna, nbm)

                if next_agg_id not in dist:
                    dist[next_agg_id] = {}
                if obs_id not in dist[next_agg_id]:
                    dist[next_agg_id][obs_id] = 0.0
                dist[next_agg_id][obs_id] += p_contrib

        # flatten into a list
        results = []
        for na_id in dist:
            for obs_id, prob_val in dist[na_id].items():
                if prob_val > 0:
                    results.append((na_id, obs_id, prob_val))
        return results

    # ------------------------------------------------
    # step (aggregate)
    # ------------------------------------------------
    def step(self, agg_state_id, action_id):
        """
        We pick an underlying state uniformly from the aggregator cell,
        do one step in the base simulator, and then map back to aggregator.

        Return (next_agg_state, obs_id, cost, done).
        """
        xa, ya, bm = self.decode_agg_state(agg_state_id)
        positions = self.aggregator_to_underlying_positions(xa, ya)
        if len(positions) == 0:
            # degenerate case: no underlying states => just stay
            return agg_state_id, 0, 0.0, True

        (ux, uy) = random.choice(positions)
        s_id = self.base_sim.encode_state(ux, uy, bm)

        # step in base
        next_s_id, obs_id, cost, done = self.base_sim.step(s_id, action_id)

        (nx, ny, nbm) = self.base_sim.decode_state(next_s_id)
        xna, yna = self.aggregator_position(nx, ny)
        next_agg_id = self.encode_agg_state(xna, yna, nbm)

        return next_agg_id, obs_id, cost, done

    # ------------------------------------------------
    # sample_next_state (aggregate)
    # ------------------------------------------------
    def sample_next_state(self, agg_state_id, action_id):
        """
        Sample a next aggregator state from the distribution
        next_state_obs_distribution(agg_state_id, action_id),
        ignoring observation.
        """
        dist_list = self.next_state_obs_distribution(agg_state_id, action_id)
        # Sum probabilities over obs
        dist_dict = {}
        for (na_id, obs_id, p) in dist_list:
            dist_dict[na_id] = dist_dict.get(na_id, 0.0) + p

        na_ids = list(dist_dict.keys())
        probs = np.array([dist_dict[na] for na in na_ids], dtype=float)
        total_p = probs.sum()
        if total_p < 1e-15:
            # fallback
            return agg_state_id
        probs /= total_p
        idx = np.random.choice(len(na_ids), p=probs)
        return na_ids[idx]

    # ------------------------------------------------
    # observation_probability (aggregate)
    # ------------------------------------------------
    def observation_probability(self, agg_state_id, action_id, obs_id):
        """
        P(obs_id | agg_state_id, action_id) by averaging base-sim obs probabilities
        over all underlying states in aggregator cell.
        """
        xa, ya, bm = self.decode_agg_state(agg_state_id)
        positions = self.aggregator_to_underlying_positions(xa, ya)
        if len(positions) == 0:
            return 0.0
        p_total = 0.0
        for (ux, uy) in positions:
            s_id = self.base_sim.encode_state(ux, uy, bm)
            p_total += self.base_sim.observation_probability(s_id, action_id, obs_id)
        return p_total / len(positions)

    # ------------------------------------------------
    # get_reward (aggregate)
    # ------------------------------------------------
    def get_reward(self, s_agg, a_id, s_next_agg=None, obs_id=None):
        """
        Return the *expected* immediate REWARD for the aggregator transition (s_agg, a, s_next_agg, obs_id).

        If s_next_agg, obs_id are provided, we compute:

            R_agg = E[ reward(s,a,s',obs') | aggregator(s)=s_agg, aggregator(s')=s_next_agg, obs'=obs_id, a ]

        If s_next_agg, obs_id are None, we compute the unconditional expected reward:

            R_agg = E[ reward(s,a,s',obs') | aggregator(s)=s_agg, a ]

        For many algorithms, you only need the unconditional expectation
        of reward from state+action, but we provide the conditional form
        for full parallel to the base simulator.
        """
        xa, ya, bm = self.decode_agg_state(s_agg)
        positions = self.aggregator_to_underlying_positions(xa, ya)
        if len(positions) == 0:
            return 0.0

        # If we have s_next_agg, obs_id, we condition on that event:
        if s_next_agg is not None and obs_id is not None:
            # Probability aggregator -> aggregator (and obs) = sum_{s in cell} p_{cell}(s) * p(s_next_agg, obs_id | s,a)
            # Then expected reward is sum_{s in cell} [ 1/|cell| * sum_{s',obs'} I(...) p(...) reward(...) ] / that prob
            dist_list = []
            # We'll accumulate numerator = E[ reward * I(event) ] and denominator = P(event)
            event_prob = 0.0
            reward_sum = 0.0

            for (ux, uy) in positions:
                s_id = self.base_sim.encode_state(ux, uy, bm)
                # Probability of choosing this s is uniform in aggregator cell => 1/|cell|
                p_s = 1.0 / len(positions)
                nx_obs_list = self.base_sim.next_state_obs_distribution(s_id, a_id)
                for (ns_id, obs_base, p_trans) in nx_obs_list:
                    if obs_base == obs_id:
                        (nx, ny, nbm) = self.base_sim.decode_state(ns_id)
                        (xna, yna) = self.aggregator_position(nx, ny)
                        agg_next_id = self.encode_agg_state(xna, yna, nbm)
                        if agg_next_id == s_next_agg:
                            # This event matches (s_next_agg, obs_id)
                            r = self.base_sim.get_reward(s_id, a_id, ns_id, obs_base)
                            event_prob += p_s * p_trans
                            reward_sum += p_s * p_trans * r

            if event_prob < 1e-15:
                return 0.0
            return reward_sum / event_prob

        else:
            # Unconditional expected reward from aggregator state + action
            # sum_{s in cell} [ 1/|cell| * sum_{s',obs'} p(s',obs'|s,a)* reward(s,a,s',obs') ]
            total_rew = 0.0
            for (ux, uy) in positions:
                s_id = self.base_sim.encode_state(ux, uy, bm)
                p_s = 1.0 / len(positions)
                ns_obs_list = self.base_sim.next_state_obs_distribution(s_id, a_id)
                local_sum = 0.0
                for (ns_id, obs_base, p_trans) in ns_obs_list:
                    r = self.base_sim.get_reward(s_id, a_id, ns_id, obs_base)
                    local_sum += p_trans * r
                total_rew += p_s * local_sum
            return total_rew

    # ------------------------------------------------
    # belief conversions
    # ------------------------------------------------
    def original_to_aggregate_belief(self, b_original):
        """
        Sum the probability of all original states that map to
        the same (xa, ya, bitmask).
        """
        b_agg = np.zeros(self.num_agg_states, dtype=float)
        for s_id, p_val in enumerate(b_original):
            if p_val == 0.0:
                continue
            x, y, bm = self.base_sim.decode_state(s_id)
            xa, ya = self.aggregator_position(x, y)
            agg_id = self.encode_agg_state(xa, ya, bm)
            b_agg[agg_id] += p_val

        total = b_agg.sum()
        if total > 1e-15:
            b_agg /= total
        return b_agg

    def aggregate_to_original_belief(self, b_agg):
        """
        Distribute each aggregator state's probability
        uniformly among all original states in that cell.
        """
        b_original = np.zeros(self.base_sim.num_states, dtype=float)
        for agg_id, p_val in enumerate(b_agg):
            if p_val == 0.0:
                continue
            (xa, ya, bm) = self.decode_agg_state(agg_id)
            positions = self.aggregator_to_underlying_positions(xa, ya)
            if len(positions) == 0:
                continue
            p_each = p_val / len(positions)
            for (x, y) in positions:
                s_id = self.base_sim.encode_state(x, y, bm)
                b_original[s_id] += p_each

        total = b_original.sum()
        if total > 1e-15:
            b_original /= total
        return b_original

    # ------------------------------------------------
    # belief_operator (exact)
    # ------------------------------------------------
    def belief_operator(self, b_agg, a_id, z_id):
        """
        b'(s'_agg) = (1/Z) * sum_{s_agg} [ b_agg[s_agg] * P(s'_agg, z_id | s_agg, a_id) ].

        We'll get the distribution from next_state_obs_distribution(s_agg, a_id).
        """
        b_prime = np.zeros(self.num_agg_states, dtype=float)
        for s_agg, p_s in enumerate(b_agg):
            if p_s == 0.0:
                continue
            dist_nsobs = self.next_state_obs_distribution(s_agg, a_id)
            for (ns_agg, obs, p_trans) in dist_nsobs:
                if obs == z_id:
                    b_prime[ns_agg] += p_s * p_trans

        norm = b_prime.sum()
        if norm < 1e-15:
            # zero-likelihood => fallback or raise
            raise ValueError("Zero-likelihood observation in aggregator belief_operator.")
        b_prime /= norm
        return b_prime

    # ------------------------------------------------
    # observation_probability (extended)
    # ------------------------------------------------
    def observation_probability_agg(self, s_agg, a_id, z_id):
        """
        P(z_id | s_agg, a_id) = sum_{(ns_agg, obs)} p(ns_agg, obs| s_agg, a_id) * I(obs==z_id).
        Or just aggregator version of self.observation_probability(...)
        using next_state_obs_distribution.
        """
        dist_nsobs = self.next_state_obs_distribution(s_agg, a_id)
        return sum(p for (ns, obs, p) in dist_nsobs if obs == z_id)

    # ------------------------------------------------
    # sample_next_state (PF usage)
    # ------------------------------------------------
    def sample_next_state_agg(self, s_agg, a_id):
        """
        Similar to the original sample_next_state, but for aggregator states.
        """
        return self.sample_next_state(s_agg, a_id)

    # ------------------------------------------------
    # belief_operator_pf (particle filter)
    # ------------------------------------------------
    def belief_operator_pf(self, b_agg, a_id, z_id, num_particles=200, eps=1e-12):
        """
        Particle filter update in the AGGREGATE simulator, implemented
        by converting the aggregator belief into the original state space,
        performing the PF update there, then converting back to the aggregator.

        :param b_agg: Current aggregator belief (NumPy array of length self.num_agg_states).
        :param a_id: Action ID (integer).
        :param z_id: Observation ID (integer).
        :param num_particles: Number of particles to use in the PF step.
        :param eps: Threshold for detecting near-zero total weight.
        :return: Updated aggregator belief (NumPy array of length self.num_agg_states).
        """
        # 1) Convert aggregator belief -> original
        b_original = self.aggregate_to_original_belief(b_agg)

        # 2) Do the particle-filter update in the original simulator
        b_original_updated = self.base_sim.belief_operator_pf(
            b=b_original,
            a_id=a_id,
            z_id=z_id,
            num_particles=num_particles,
            eps=eps
        )

        # 3) Convert the updated original belief -> aggregator
        b_agg_new = self.original_to_aggregate_belief(b_original_updated)

        return b_agg_new
    # ------------------------------------------------
    # precompute_cost
    # ------------------------------------------------
    def precompute_cost(self):
        """
        cost_xu[a, s_agg] = sum_{(s'_agg, obs)} P(s'_agg,obs| s_agg,a)* [ - get_reward(s_agg,a,s'_agg,obs) ].
        """
        cost_xu = np.zeros((self.num_actions, self.num_agg_states), dtype=float)
        for a_id in range(self.num_actions):
            for s_agg in range(self.num_agg_states):
                total_cost = 0.0
                dist_nsobs = self.next_state_obs_distribution(s_agg, a_id)
                for (ns_agg, obs_id, p_trans) in dist_nsobs:
                    r = self.get_reward(s_agg, a_id, ns_agg, obs_id)
                    total_cost += p_trans * (-r)
                cost_xu[a_id, s_agg] = total_cost
        return cost_xu

    def initial_belief(self):
        """
        Returns a probability vector (length = self.num_agg_states)
        over aggregator states where the aggregator cell corresponds
        to the cell containing (0,0), and bitmask is uniform.

        That is:
            (x_a, y_a) = aggregator_position(0,0),
            bitmask in [0..2^k - 1] each with probability 1/(2^k).
        """
        b_agg = np.zeros(self.num_agg_states, dtype=float)

        # Which aggregator cell contains the original (0,0)?
        (xa0, ya0) = self.aggregator_position(0, 0)

        # Assign uniform probability across all bitmasks in that cell
        for bitmask in range(2 ** self.k):
            agg_state_id = self.encode_agg_state(xa0, ya0, bitmask)
            b_agg[agg_state_id] = 1.0

        # Normalize
        b_agg /= b_agg.sum()
        return b_agg

    def init_state(self):
        """
        Returns a single aggregator state ID where
        the aggregator cell corresponds to the cell containing (0,0),
        and the rock bitmask is sampled randomly (each rock good w.p. 0.5).
        """
        (xa0, ya0) = self.aggregator_position(0, 0)

        # Sample bitmask
        bitmask = 0
        for i in range(self.k):
            if random.random() < 0.5:
                bitmask |= (1 << i)

        return self.encode_agg_state(xa0, ya0, bitmask)

    # ------------------------------------------------
    # policy_evaluation
    # ------------------------------------------------
    def policy_evaluation(self, L, mu, gamma, N, b_agg_to_index=None, n=4, use_pf=False):
        """
        Evaluate a policy mu over aggregator states for L episodes,
        each up to N steps, starting from aggregator belief b0_agg.
        mu is presumably of shape [|B_agg|, num_actions].
        """
        costs = []
        avg_cost = 0.0

        for l in range(L):
            s_id = self.base_sim.init_state()
            b = self.base_sim.initial_belief()
            total_cost = 0.0

            for step_idx in range(N):
                b_agg = self.original_to_aggregate_belief(b_original=b)
                # find the policy action from aggregator belief
                if b_agg_to_index is not None:
                    if tuple(b_agg) in b_agg_to_index:
                        b_idx = b_agg_to_index[tuple(b_agg)]
                    else:
                        b_rounded = POMDPUtil.round_to_Bn(b_agg, n=n)
                        b_idx = b_agg_to_index[tuple(b_rounded)]
                    a_id = np.argmax(mu[b_idx])
                else:
                    a_id = np.random.randint(self.num_actions)

                # environment step
                next_s_id, obs_id, cost, done = self.step(s_id, a_id)
                total_cost += (gamma ** step_idx) * cost

                # belief update
                if use_pf:
                    b = self.belief_operator_pf(b, a_id, obs_id, num_particles=50)
                else:
                    b = self.belief_operator(b, a_id, obs_id)

                s_agg = next_s_id
                if done:
                    break

            costs.append(total_cost)
            avg_cost = np.mean(costs)
        return avg_cost


###############################################################################
# Example main
###############################################################################
if __name__ == "__main__":
    # Create the original simulator
    base_env = RockSampleSimulator(n=4, k=2, seed=123)
    print("Base simulator: #states =", base_env.num_states)

    # Precompute cost in the original simulator
    cost_xu = base_env.precompute_cost()
    print("Original precomputed cost shape:", cost_xu.shape)

    # Create aggregator
    agg_env = AggregateRockSampleSimulator(base_env, grid_resolution=2)
    print("Aggregate simulator: #states =", agg_env.num_agg_states)

    # Precompute aggregator cost
    cost_xu_agg = agg_env.precompute_cost()
    print("Aggregate precomputed cost shape:", cost_xu_agg.shape)

    # Evaluate a random policy in the original simulator for demonstration
    b0 = base_env.initial_belief()
    mu_random = np.random.rand(1, base_env.num_actions)  # a trivial policy for demonstration
    mu_random /= mu_random.sum(axis=1, keepdims=True)
    # We'll do a dictionary for b->index with just one "index" = 0
    b_to_index = {tuple(b0): 0}
    avg_cost_orig = base_env.policy_evaluation(L=3, mu=mu_random, gamma=0.95,
                                               b0=b0, N=10,
                                               b_to_index=b_to_index, use_pf=False)
    print(f"Original policy evaluation average cost = {avg_cost_orig:.3f}")

    # Evaluate a random aggregator policy
    b0_agg = agg_env.original_to_aggregate_belief(b0)
    mu_agg_random = np.random.rand(1, agg_env.num_actions)
    mu_agg_random /= mu_agg_random.sum(axis=1, keepdims=True)
    b_agg_to_index = {tuple(b0_agg): 0}
    avg_cost_agg = agg_env.policy_evaluation(L=3, mu=mu_agg_random, gamma=0.95,
                                             b0_agg=b0_agg, N=10,
                                             b_agg_to_index=b_agg_to_index, use_pf=False)
    print(f"Aggregate policy evaluation average cost = {avg_cost_agg:.3f}")
