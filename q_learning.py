import random
import math
import numpy as np


###############################################################################
# Utilities: rounding a belief to B_n, and a simple particle filter.
###############################################################################

def round_to_Bn(b, n):
    """
    Round the belief vector b (length S) into B_n, i.e. a vector b' = (k_1/n, ..., k_S/n)
    summing to 1, with each k_i integer >= 0 and sum(k_i) = n.

    Steps:
      1) Scale up: w_i = n * b_i
      2) floor_i = floor(w_i)
      3) leftover = n - sum(floor_i)
      4) Use the largest fractional parts to distribute leftover increments.
      5) b'_i = floor_i[i]/n (plus an extra 1 for those leftover slots).
    """
    b_list = list(b)
    S = len(b_list)

    # 1) scale up
    w = [x * n for x in b_list]

    # 2) floor each w_i
    floor_vals = [math.floor(x) for x in w]

    sum_floor = sum(floor_vals)
    leftover = n - sum_floor

    # fractional parts
    frac_parts = [(w[i] - floor_vals[i], i) for i in range(S)]
    frac_parts.sort(key=lambda x: x[0], reverse=True)

    # 3) distribute leftover
    for k in range(leftover):
        i_coord = frac_parts[k][1]
        floor_vals[i_coord] += 1

    # 4) convert to final b' by dividing by n
    b_rounded = [fv / n for fv in floor_vals]

    return b_rounded


def particle_filter(
        old_belief,
        action_id,
        obs_id,
        env,
        num_particles=100
):
    """
    Particle filter step using 'rejection sampling'.

    :param old_belief: A distribution (list of length S) over the states.
    :param action_id: Chosen action in integer form.
    :param obs_id: Observed observation (integer).
    :param env: Environment (must have env.step(state, action_id)).
    :param num_particles: Number of particles to keep in the new distribution.

    :return: A distribution (list of length S) that sums to 1,
             consistent with (action_id, obs_id).
    """
    S = env.num_states
    new_counts = [0] * S
    accepted = 0

    # For sampling states from old_belief
    states = list(range(S))

    while accepted < num_particles:
        x = np.random.choice(states, p=old_belief)
        x_next, next_obs, _, _ = env.step(x, action_id)
        if next_obs == obs_id:
            new_counts[x_next] += 1
            accepted += 1

    # Convert counts to a normalized distribution
    new_belief = [cnt / float(num_particles) for cnt in new_counts]
    return new_belief


###############################################################################
# Q-Learning in B_n, with periodic policy evaluation
###############################################################################

def evaluate_greedy_policy(Q, env, B_n, b_to_id, gamma, n, num_episodes=100, max_steps=50, start_belief_id=0):
    """
    Evaluate the discounted return of the greedy policy w.r.t. Q over multiple episodes.

    :param Q: Q-table dict -> Q[(b_id, a)] = value
    :param env: The environment (RockSampleInt or similar).
    :param B_n: List of discrete beliefs (each an array of length S).
    :param b_to_id: Dict mapping each belief (as a tuple) to its index in B_n.
    :param gamma: Discount factor
    :param num_episodes: Number of episodes for the evaluation
    :param max_steps: Maximum steps per episode
    :param start_belief_id: The index of the initial belief in B_n (for all episodes).
      Alternatively, you could randomize the starting belief or do something else.

    :return: Average discounted return across the episodes.
    """
    S = env.num_states
    A = env.num_actions

    def get_q(b_id, a):
        return Q.get((b_id, a), 0.0)

    total_return = 0.0

    for _ in range(num_episodes):
        # Start from the chosen belief
        b_id = start_belief_id
        b_current = B_n[b_id]

        episode_return = 0.0
        discount = 1.0

        for _step in range(max_steps):
            # Greedy action: argmax_a Q(b, a)
            q_values = [get_q(b_id, a_) for a_ in range(A)]
            a = int(np.argmax(q_values))

            # Sample a state from b_current
            x = np.random.choice(range(S), p=b_current)
            x_next, obs, reward, done = env.step(x, a)

            # Accumulate discounted reward
            episode_return += discount * reward
            discount *= gamma

            if done:
                # End the episode
                break
            # Otherwise, do the particle filter
            b_next_raw = particle_filter(b_current, a, obs, env, num_particles=50)
            # Round to B_n
            b_next_rounded = round_to_Bn(b_next_raw, n)  # or some other n
            b_next_key = tuple(b_next_rounded)
            if b_next_key in b_to_id:
                b_id = b_to_id[b_next_key]
            else:
                # If not found, pick the best match in B_n (or add it).
                b_id = min(range(len(B_n)),
                           key=lambda i: sum(abs(B_n[i][s] - b_next_rounded[s]) for s in range(S)))

            b_current = B_n[b_id]

        total_return += episode_return

    avg_return = total_return / float(num_episodes)
    return avg_return


def run_q_learning_in_Bn(env, B_n, n, gamma=0.95, alpha=0.1, epsilon=0.1,
                         num_iterations=50, log_frequency=10, start_belief_id=0):
    """
    Q-learning in a discrete belief set B_n. Logs the average discounted reward of
    the greedy policy every 'log_frequency' iterations.

    :param env: The environment (e.g. RockSampleInt).
    :param B_n: List of discrete beliefs, each of length S, summing to 1.0
    :param gamma: Discount factor
    :param alpha: Learning rate
    :param epsilon: Epsilon for epsilon-greedy
    :param num_iterations: number of Q-learning updates
    :param log_frequency: how many iterations between policy evaluation
    :param start_belief_id: index of the initial belief in B_n for the Q-learning episodes

    :return: The learned Q-table (dict) and the final policy evaluation result.
    """
    S = env.num_states
    A = env.num_actions
    O = env.num_observations

    # We'll store Q-values in Q[(b_id, a)]
    Q = {}

    def get_q(b_id, a):
        return Q.get((b_id, a), 0.0)

    def set_q(b_id, a, val):
        Q[(b_id, a)] = val

    # Create a mapping from beliefs (as a tuple) -> index
    b_to_id = {tuple(b): i for i, b in enumerate(B_n)}

    # Start from the given initial belief
    b_id = start_belief_id
    b_current = B_n[b_id]

    for it in range(num_iterations):
        # Epsilon-greedy action
        if random.random() < epsilon:
            a = random.randint(0, A - 1)
        else:
            # pick best action from Q
            values = [get_q(b_id, a_) for a_ in range(A)]
            a = int(np.argmax(values))

        # 1) Sample a state from b_current
        x = np.random.choice(range(S), p=b_current)
        # 2) Step environment
        x_next, obs, reward, done = env.step(x, a)

        # 3) If not done, build next belief
        if not done:
            b_next_raw = particle_filter(b_current, a, obs, env, num_particles=50)
            # 4) Round b_next_raw to remain in B_n
            b_next_rounded = round_to_Bn(b_next_raw, n)  # or some n for B_n
            b_next_key = tuple(b_next_rounded)
            if b_next_key in b_to_id:
                b_next_id = b_to_id[b_next_key]
            else:
                # e.g. pick nearest in B_n
                b_next_id = min(range(len(B_n)),
                                key=lambda i: sum(abs(B_n[i][s] - b_next_rounded[s]) for s in range(S)))

            # Q-learning update
            old_q = get_q(b_id, a)
            max_next_q = max(get_q(b_next_id, a_) for a_ in range(A))
            target = reward + gamma * max_next_q
            new_q = old_q + alpha * (target - old_q)
            set_q(b_id, a, new_q)

            # Move to next belief
            b_id = b_next_id
            b_current = B_n[b_id]
        else:
            # Terminal: do the update with no next state
            old_q = get_q(b_id, a)
            target = reward
            new_q = old_q + alpha * (target - old_q)
            set_q(b_id, a, new_q)

            # Reset to the initial belief for the next iteration
            b_id = start_belief_id
            b_current = B_n[b_id]

        # ---- Periodic policy evaluation ----
        if (it + 1) % log_frequency == 0:
            avg_discounted = evaluate_greedy_policy(
                Q, env, B_n, b_to_id, gamma, n=n,
                num_episodes=10, max_steps=100,
                start_belief_id=start_belief_id
            )
            print(f"Iteration {it + 1}: Average discounted return of the "
                  f"greedy policy = {avg_discounted:.4f} Q: {sum(Q.values()):.4f}")

    # Final evaluation after training
    final_eval = evaluate_greedy_policy(
        Q, env, B_n, b_to_id, gamma, n=n, num_episodes=100, max_steps=50,
        start_belief_id=start_belief_id
    )
    print(f"Final average discounted return: {final_eval:.4f}, Q: {sum(Q.values()):.4f}")

    return Q, final_eval


###############################################################################
# Example usage (with a mock environment):
###############################################################################

def main():
    # In your real code, you'd import your RockSampleInt environment and construct it like:
    #
    #   env = RockSampleInt(n=4, k=4, seed=999)
    #   env.kBeliefN = 5  # or some integer used by round_to_Bn
    #
    # For demonstration, we'll define a small mock environment with 2 states, etc.

    class MockEnv:
        def __init__(self):
            self.num_states = 2
            self.num_actions = 2
            self.num_observations = 2
            self.kBeliefN = 2  # used by round_to_Bn

        def step(self, state, action):
            # trivial example: flips state, obs=0 always, reward=1 if next_state=1
            next_state = 1 - state
            obs = 0
            reward = 1 if next_state == 1 else 0
            done = (action == 1)  # end if action=1
            return next_state, obs, reward, done

    env = MockEnv()

    # Construct B_n for n=2 => possible beliefs are [1,0], [0.5,0.5], [0,1]
    B_n = [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ]

    Q_table, final_eval = run_q_learning_in_Bn(
        env,
        B_n,
        gamma=0.95,
        alpha=0.1,
        epsilon=0.1,
        num_iterations=500,
        log_frequency=1,
        start_belief_id=0
    )

    print("\n=== Training Complete ===")
    print("Final Q-table size:", len(Q_table))
    print("Final policy eval:", final_eval)


if __name__ == "__main__":
    main()
