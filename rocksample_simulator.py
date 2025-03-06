#!/usr/bin/env python3

import random
import numpy as np
from pomdp_util import POMDPUtil


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
        Returns an initial state with x=0, y=0, each rock good/bad w.p. 0.5
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
        Returns (next_state_id, obs_id, reward, done).
        NOTE: the final returned 'reward' is actually -reward internally
              because this simulator uses 'reward += ...' then returns '-reward'.
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
            # else remain in place (or do nothing)

        elif action_str == 'south':
            nx, ny = x, y - 1
            if self.valid_position(nx, ny):
                x, y = nx, ny

        elif action_str == 'east':
            nx, ny = x + 1, y
            if self.valid_position(nx, ny):
                x, y = nx, ny
            else:
                # possibly exit
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

            # If we did not find any rock at (x,y), impose -100 penalty
            if not rock_found:
                reward -= 100

        elif action_str.startswith('sense_'):
            # parse the rock index
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

        # Build next state
        next_state_id = self.encode_state(x, y, bitmask)

        # If we stepped off the east edge exactly:
        if (not done) and (x == self.n):
            done = True

        # Return -reward as the final "reported" reward
        # i.e. if reward is +10 internally, the agent sees -10 cost, etc.
        return next_state_id, obs_id, -reward, done

    # ----------------------------
    # Next-state/obs distribution
    # ----------------------------
    def next_state_obs_distribution(self, state_id, action_id):
        """
        Return a list of (next_state_id, obs_id, probability)
        enumerating all possible next states & observations
        from (state_id, action_id).

        - For moves/sample: there's exactly 1 next state, 1 possible obs='none' => prob=1.
        - For sense_i: exactly 1 next state,
          but obs can be 'good' or 'bad' with some probability
          depending on distance & rock's actual bit.
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
                # typically at x==n => done, but let's clamp for distribution
                pass

        elif a_str == 'west':
            maybe_x = x - 1
            if self.valid_position(maybe_x, y):
                nx = maybe_x

        elif a_str == 'sample':
            # If on a rock, we clear that rock's bit
            for i, (rx, ry) in enumerate(self.rock_positions):
                if rx == x and ry == y:
                    nbitmask = nbitmask & ~(1 << i)
                    break
            # Note: does not handle "invalid sample" penalty in the distribution
            # but it's the same next state. Observations are always 'none'.

        elif a_str.startswith('sense_'):
            # next state is same (except for potential out-of-bounds),
            # observation distribution is separate
            pass
        else:
            raise ValueError(f"Unknown action '{a_str}'")

        # Clamp or check bounds
        if not (0 <= nx < self.n) or not (0 <= ny < self.n):
            nx = max(0, min(nx, self.n - 1))
            ny = max(0, min(ny, self.n - 1))

        next_s_id = self.encode_state(nx, ny, nbitmask)

        # 2) Define obs distribution
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
            # all other actions => single obs = 'none'
            obs_none = self.obs_str2id['none']
            possible_obs.append((obs_none, 1.0))

        # 3) Build final distribution => list of (next_state_id, obs_id, probability)
        result = []
        for (obs_id, p) in possible_obs:
            if p > 0.0:
                result.append((next_s_id, obs_id, p))

        return result

    # -----------------------------
    # Convenience methods
    # -----------------------------
    def action_str_to_id(self, a_str):
        return self.action_str2id[a_str]

    def action_id_to_str(self, a_id):
        return self.action_id2str[a_id]

    def obs_str_to_id(self, o_str):
        return self.obs_str2id[o_str]

    def obs_id_to_str(self, o_id):
        return self.obs_id2str[o_id]

    # ----------------------------
    # Belief Update
    # ----------------------------
    def belief_operator(self, b, a_id, z_id):
        """
        Update the belief distribution b (over env.num_states) given that:
          - the agent took action a_id,
          - an observation z_id was received.

        b'(x') = (1 / norm) * sum_{x} [ b[x] * P(x', z | x, a) ]
               = (1 / norm) * sum_{x} [ b[x] * P(x'|x, a) * P(z|x', a) ]

        Uses next_state_obs_distribution(...) for enumerating transitions & observation probs.
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
        if norm <= 0.0:
            raise ValueError(f"Zero-likelihood observation {z_id} for belief update.")

        b_prime /= norm

        # optional check
        if not np.allclose(b_prime.sum(), 1.0, rtol=1e-7, atol=1e-9):
            raise ValueError(f"Belief does not sum to 1 after update (sum={b_prime.sum():.6f}).")

        return b_prime

    def observation_probability(self, x_id, a_id, z_id):
        """
        Return P(z_id | x_id, a_id).
        For move/sample actions: the observation is always 'none' (id=0).
        For sense_i actions: we compute the probability that the observation is 'good' or 'bad'.
        """
        a_str = self.action_id2str[a_id]
        # Decode the state:
        x, y, bitmask = self.decode_state(x_id)

        # --- If it's a sense action ---
        if a_str.startswith('sense_'):
            i_rock = int(a_str.split('_')[1])
            rx, ry = self.rock_positions[i_rock]
            dist = abs(x - rx) + abs(y - ry)
            correct_prob = 1.0 - (0.5 ** dist)
            # Check if rock i is good or bad in this state:
            rock_good = ((bitmask & (1 << i_rock)) != 0)

            if rock_good:
                # Probability z='good' is correct_prob, z='bad' is 1-correct_prob
                if z_id == self.obs_str2id['good']:
                    return correct_prob
                elif z_id == self.obs_str2id['bad']:
                    return 1.0 - correct_prob
                else:
                    return 0.0
            else:
                # Rock is bad
                if z_id == self.obs_str2id['bad']:
                    return correct_prob
                elif z_id == self.obs_str2id['good']:
                    return 1.0 - correct_prob
                else:
                    return 0.0

        # --- Otherwise, action is move or sample ---
        else:
            # For these actions, the observation is always 'none'.
            if z_id == self.obs_str2id['none']:
                return 1.0
            else:
                return 0.0

    def sample_next_state(self, x_id, a_id):
        """
        Sample a next state x' from P(x' | x_id, a_id) by
        ignoring the observation in the distribution (x', obs, p).

        Returns one integer x' (the next state).
        """
        dist_dict = {}
        # next_state_obs_distribution returns list of (nx_id, obs, p)
        for (nx_id, obs_id, p) in self.next_state_obs_distribution(x_id, a_id):
            # Accumulate probabilities for each nx_id
            if nx_id not in dist_dict:
                dist_dict[nx_id] = 0.0
            dist_dict[nx_id] += p

        # Now dist_dict[x'] = sum of probabilities for that next state
        # We do a random sample from those next-state probabilities.

        # Convert to arrays
        nx_ids = list(dist_dict.keys())
        probs = np.array([dist_dict[nx] for nx in nx_ids], dtype=float)
        # Normalize (should already sum to 1, but do it just in case of float fuzz)
        probs /= probs.sum()

        # Draw a single sample
        chosen_idx = np.random.choice(len(nx_ids), p=probs)
        return nx_ids[chosen_idx]

    def belief_operator_pf(self, b, a_id, z_id, num_particles=500, eps=1e-12):
        """
        Particle filter (PF) update for the belief distribution, robust to zero/near-zero weights.

        :param b: current belief (length = self.num_states) as a NumPy array or list
        :param a_id: the integer action ID
        :param z_id: the integer observation ID
        :param num_particles: how many particles to use in the PF update
        :param eps: small threshold for detecting near-zero total weight
        :return: a new belief distribution (length = self.num_states) as a NumPy array
        """
        # 1) Sample N particles from the current belief b
        state_ids = np.arange(self.num_states)
        particles = np.random.choice(state_ids, size=num_particles, p=b)

        # 2) For each particle, sample next state & compute weight
        next_particles = np.zeros(num_particles, dtype=int)
        weights = np.zeros(num_particles, dtype=float)

        for i in range(num_particles):
            x_cur = particles[i]
            # Sample next state from x_cur under action a
            x_next = self.sample_next_state(x_cur, a_id)
            next_particles[i] = x_next
            # Importance weight = P(z | x_next, a)
            w = self.observation_probability(x_next, a_id, z_id)
            weights[i] = w

        # 3) Normalize weights
        w_sum = weights.sum()

        if w_sum < eps:
            # All (or most) weights are zero => can't update meaningfully from this observation
            # Fallback: return uniform (or do something else if you prefer)
            b_prime = np.ones(self.num_states, dtype=float) / self.num_states
            return b_prime
        else:
            # 4) Resample according to normalized weights
            weights /= w_sum
            indices_resampled = np.random.choice(num_particles, size=num_particles, p=weights)
            resampled_particles = next_particles[indices_resampled]

            # 5) Convert the resampled particles to a discrete distribution
            b_prime = np.zeros(self.num_states, dtype=float)
            for x_p in resampled_particles:
                b_prime[x_p] += 1.0

            # 6) Normalize to ensure sum=1
            b_prime /= num_particles
            return b_prime

    # ----------------------------
    # Reward function replication
    # ----------------------------
    def get_reward(self, x_id, a_id, x_next_id, obs_id):
        """
        Return the immediate reward for (x, a, x', obs).
        We replicate the logic from step(...) but in a deterministic manner
        (i.e., no random draws).

        Because step(...) returns (-reward), we will keep that same sign usage here:
          - "rew" in code below is the simulator's internal variable
            so the actual returned value from get_reward(...) is the
            environment's immediate *reward* (not cost).
        """
        # decode old state
        x, y, bitmask = self.decode_state(x_id)
        # decode new state
        nx, ny, nbitmask = self.decode_state(x_next_id)
        a_str = self.action_id_to_str(a_id)

        # Replicate the internal logic
        rew = 0.0

        # 1) step cost
        rew += self.step_cost

        # 2) check if 'east' action might cause exit
        if a_str == 'east':
            # If next state's x == self.n => agent presumably exited
            # But we have to see if stepping from x to x+1 was out of bounds
            if (nx == x) and (x + 1 == self.n):
                # this can happen if the environment "did nothing" but recognized the exit
                # or if the environment is done.
                # We'll add the exit reward if we "exited."
                rew += self.exit_reward
            # Alternatively, if nx == self.n in a hypothetical approach,
            # you can add the exit reward. Adjust to your design.

        # 3) check 'sample' logic:
        if a_str == 'sample':
            # Replicate: we look for a rock at (x,y).
            rock_found = False
            for i, (rx, ry) in enumerate(self.rock_positions):
                if rx == x and ry == y:
                    rock_found = True
                    # If old bitmask had i set => rock was good
                    if (bitmask & (1 << i)) != 0:
                        rew += self.sample_good_reward
                    else:
                        rew += self.sample_bad_reward
                    break
            if not rock_found:
                # invalid sample => -100
                rew -= 100

        # sense_i has no direct reward other than step_cost
        # done = ?

        return rew

    # ----------------------------
    # Example: precompute cost
    # ----------------------------
    def precompute_cost(self):
        """
        Precompute the immediate *cost* for each (action, state) in the environment,
        cost_xu[a, s] = sum_{(s_next,obs)} p(s_next,obs|s,a)* [ - reward(s,a,s_next,obs) ].
        We'll store it in a 2D np.array of shape [num_actions, num_states].
        """
        cost_xu = np.zeros((self.num_actions, self.num_states), dtype=float)

        for a_id in range(self.num_actions):
            # For each action
            for s_id in range(self.num_states):
                total_cost = 0.0
                nx_dist = self.next_state_obs_distribution(s_id, a_id)  # (ns, obs, p)
                for (ns, obs, p_nsobs) in nx_dist:
                    r = self.get_reward(s_id, a_id, ns, obs)
                    cost_val = -r
                    total_cost += p_nsobs * cost_val
                cost_xu[a_id, s_id] = total_cost

        return cost_xu

    # ----------------------------
    # A basic policy evaluation
    # ----------------------------
    def policy_evaluation(self, L, mu, gamma, b0, N, B_n = None, b_to_index=None, n=4, use_pf = False):
        """
        Example function that simulates a policy mu for L episodes,
        each up to N steps, starting from belief b0.
        (mu is presumably an array of size [|B|, num_actions], maybe from a solver.)
        """
        costs = []
        avg_cost = 0.0

        for l in range(L):
            # print(f"Policy evaluation iteration: {l}/{L}, avg cost: {avg_cost}")
            s_id = np.random.choice(self.num_states, p=b0)
            b = b0.copy()
            total_cost = 0.0

            for step_idx in range(N):
                # Identify which belief index we are at
                if tuple(b) in b_to_index:
                    b_idx = b_to_index[tuple(b)]
                else:
                    # b_idx = POMDPUtil.nearest_neighbor(B_n=B_n, b=b)[1]
                    b_rounded = POMDPUtil.round_to_Bn(b, n=n)
                    b_idx = b_to_index[tuple(b_rounded)]

                # Choose action with highest value in mu[b_idx]
                a_id = np.argmax(mu[b_idx])

                # environment step
                next_s_id, obs_id, cost, done = self.step(s_id, a_id)
                total_cost += (gamma ** step_idx) * (cost)

                # belief update
                if use_pf:
                    b = self.belief_operator_pf(b=b, a_id=a_id, z_id=obs_id, num_particles=50)
                else:
                    b = self.belief_operator(b=b, a_id=a_id, z_id=obs_id)

                s_id = next_s_id
                if done:
                    break

            costs.append(total_cost)
            avg_cost = np.mean(costs) if len(costs) > 0 else 0.0

        return avg_cost


def main(n, k, seed=999):
    env = RockSampleSimulator(n=n, k=k, seed=seed)
    print("Number of states:", env.num_states)
    print("Number of actions:", env.num_actions)
    print("Number of observations:", env.num_observations)
    print("Rock positions:", env.rock_positions)

    # Let's pick a random state:
    s_id = env.init_state()
    print("Random initial state:", s_id, "=>", env.decode_state(s_id))

    # Choose an action, say sense_0 if k>0 else 'sample'
    if k > 0:
        a_str = "sense_0"
    else:
        a_str = "sample"
    a_id = env.action_str_to_id(a_str)

    dist = env.next_state_obs_distribution(s_id, a_id)
    print(f"\nNext-state & obs distribution from state {s_id} ({env.decode_state(s_id)}) "
          f"with action {a_id} ({a_str}):")
    for (ns, obs, prob) in dist:
        print(f"   next_state={ns} => {env.decode_state(ns)}, "
              f"obs={obs} ({env.obs_id_to_str(obs)}), p={prob:.3f}")


if __name__ == "__main__":
    main(n=4, k=3, seed=999)
