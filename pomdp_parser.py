"""
parse_pomdp.py

An extended parser for .POMDP files that:
  - Collects the usual fields (discount, states, actions, observations).
  - Stores T/O/R in dict-of-dicts-of-dicts form.
  - Also builds 0-indexed tensors T, Z, and R (nested lists).
  - Provides X (the list of valid state indices),
    O (the list of valid observation indices),
    plus lookup dicts for states, actions, and observations.
"""

import re

class POMDPParser:
    """
    A container for POMDP specifications and derived data:
      - discount: float
      - values: typically 'reward'
      - states, actions, observations: string labels
      - transition_probs[action][s_from][s_to]
      - observation_probs[action][s_to][obs]
      - rewards[action][s_from][s_to][obs]

    Additionally, it holds:
      - state_name_to_id, state_id_to_name
      - action_name_to_id, action_id_to_name
      - obs_name_to_id,    obs_id_to_name

      - X: list of state indices (0..N-1)
      - O: list of observation indices (0..M-1)

      - T[a][s][s']: transition probabilities (nested lists)
      - Z[a][s'][o]: observation probabilities (nested lists)
      - R[a][s][s'][o]: rewards (nested lists)
    """

    def __init__(self):
        # Basic fields
        self.discount = None
        self.values = None
        self.states = []
        self.actions = []
        self.observations = []

        # Dict-based probabilities and rewards
        # e.g. transition_probs[action][state_from][state_to] = prob
        self.transition_probs = {}
        # observation_probs[action][state_to][observation] = prob
        self.observation_probs = {}
        # rewards[action][state_from][state_to][observation] = value
        self.rewards = {}

        # Index-based lookups
        self.state_name_to_id = {}
        self.state_id_to_name = {}
        self.action_name_to_id = {}
        self.action_id_to_name = {}
        self.obs_name_to_id = {}
        self.obs_id_to_name = {}

        # Numeric sets (lists) of indices
        self.X = []  # states
        self.O = []  # observations
        self.A = []  # actions

        # Tensor-like nested lists
        self.T = []  # T[a][s][s']
        self.Z = []  # Z[a][s'][o]
        self.R = []  # R[a][s][s'][o]

    def initialize_transitions(self):
        """Initialize transition_probs with zeros for each (action, s_from, s_to)."""
        for a in self.actions:
            self.transition_probs[a] = {}
            for s_from in self.states:
                self.transition_probs[a][s_from] = {}
                for s_to in self.states:
                    self.transition_probs[a][s_from][s_to] = 0.0

    def initialize_observations(self):
        """Initialize observation_probs with zeros for each (action, s_to, obs)."""
        for a in self.actions:
            self.observation_probs[a] = {}
            for s_to in self.states:
                self.observation_probs[a][s_to] = {}
                for obs in self.observations:
                    self.observation_probs[a][s_to][obs] = 0.0

    def initialize_rewards(self):
        """Initialize rewards with zeros for each (action, s_from, s_to, obs)."""
        for a in self.actions:
            self.rewards[a] = {}
            for s_from in self.states:
                self.rewards[a][s_from] = {}
                for s_to in self.states:
                    self.rewards[a][s_from][s_to] = {}
                    for obs in self.observations:
                        self.rewards[a][s_from][s_to][obs] = 0.0

    @staticmethod
    def parse_pomdp(file_path):
        """
        Parse a .POMDP file and return a POMDP object with:
          - discount, values, states, actions, observations
          - transition_probs, observation_probs, rewards (dict-of-dicts)
          - T, Z, R as nested Python lists (indexed from 0)
          - X, O as the lists [0..num_states-1], [0..num_obs-1]
          - Lookup dicts (state_name_to_id, etc.)
        """
        pomdp = POMDPParser()

        # Read non-empty, non-comment lines
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

        # Regex for basic fields
        discount_pattern = re.compile(r"^discount\s*:\s*([\d\.]+)")
        values_pattern = re.compile(r"^values\s*:\s*(\S+)")
        states_pattern = re.compile(r"^states\s*:\s*(.*)")
        actions_pattern = re.compile(r"^actions\s*:\s*(.*)")
        obs_pattern = re.compile(r"^observations\s*:\s*(.*)")

        # Regex for T/O lines: e.g. T: action ...
        transition_pattern = re.compile(r"^T\s*:\s*([^:]+)(.*)")
        observation_pattern = re.compile(r"^O\s*:\s*([^:]+)(.*)")

        # Regex for R lines: e.g. R: action : s_from : s_to : obs value
        reward_pattern = re.compile(
            r"^R\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s+([-]?\d+(\.\d+)?)"
        )

        # 1) First pass: gather discount, values, states, actions, observations
        i = 0
        while i < len(lines):
            line = lines[i]
            if discount_pattern.match(line):
                pomdp.discount = float(discount_pattern.match(line).group(1))
                i += 1
                continue

            if values_pattern.match(line):
                pomdp.values = values_pattern.match(line).group(1)
                i += 1
                continue

            if states_pattern.match(line):
                # e.g. "states: tiger-left tiger-right"
                pomdp.states = states_pattern.match(line).group(1).split()
                i += 1
                continue

            if actions_pattern.match(line):
                pomdp.actions = actions_pattern.match(line).group(1).split()
                i += 1
                continue

            if obs_pattern.match(line):
                pomdp.observations = obs_pattern.match(line).group(1).split()
                i += 1
                continue

            i += 1  # skip lines that don't match these patterns

        # Initialize T/O/R dicts
        pomdp.initialize_transitions()
        pomdp.initialize_observations()
        pomdp.initialize_rewards()

        # 2) Second pass: parse T, O, R
        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Transition lines ---
            tm = transition_pattern.match(line)
            if tm:
                action = tm.group(1).strip()
                suffix = tm.group(2).strip()  # could be 'identity', 'uniform', or empty

                if not suffix:  # maybe on next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('identity'):
                            for s_from in pomdp.states:
                                for s_to in pomdp.states:
                                    pomdp.transition_probs[action][s_from][s_to] = 1.0 if s_from == s_to else 0.0
                            i += 2
                            continue
                        elif next_line.startswith('uniform'):
                            nS = len(pomdp.states)
                            for s_from in pomdp.states:
                                for s_to in pomdp.states:
                                    pomdp.transition_probs[action][s_from][s_to] = 1.0 / nS
                            i += 2
                            continue
                    # Otherwise, assume a matrix follows
                    i += 1
                    for s_index, s_from in enumerate(pomdp.states):
                        prob_line = lines[i].split()
                        i += 1
                        for t_index, s_to in enumerate(pomdp.states):
                            pomdp.transition_probs[action][s_from][s_to] = float(prob_line[t_index])
                    continue

                # If suffix is on the same line
                if suffix.startswith("identity"):
                    for s_from in pomdp.states:
                        for s_to in pomdp.states:
                            pomdp.transition_probs[action][s_from][s_to] = 1.0 if s_from == s_to else 0.0
                    i += 1
                    continue
                elif suffix.startswith("uniform"):
                    nS = len(pomdp.states)
                    for s_from in pomdp.states:
                        for s_to in pomdp.states:
                            pomdp.transition_probs[action][s_from][s_to] = 1.0 / nS
                    i += 1
                    continue
                else:
                    # parse matrix
                    i += 1
                    for s_index, s_from in enumerate(pomdp.states):
                        prob_line = lines[i].split()
                        i += 1
                        for t_index, s_to in enumerate(pomdp.states):
                            pomdp.transition_probs[action][s_from][s_to] = float(prob_line[t_index])
                    continue

            # --- Observation lines ---
            om = observation_pattern.match(line)
            if om:
                action = om.group(1).strip()
                suffix = om.group(2).strip()

                if not suffix:  # maybe on next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('identity'):
                            for s_to in pomdp.states:
                                for obs in pomdp.observations:
                                    pomdp.observation_probs[action][s_to][obs] = 1.0 if s_to == obs else 0.0
                            i += 2
                            continue
                        elif next_line.startswith('uniform'):
                            nO = len(pomdp.observations)
                            for s_to in pomdp.states:
                                for obs in pomdp.observations:
                                    pomdp.observation_probs[action][s_to][obs] = 1.0 / nO
                            i += 2
                            continue
                    # parse matrix
                    i += 1
                    for s_index, s_to in enumerate(pomdp.states):
                        prob_line = lines[i].split()
                        i += 1
                        for o_index, obs in enumerate(pomdp.observations):
                            pomdp.observation_probs[action][s_to][obs] = float(prob_line[o_index])
                    continue

                # suffix on same line
                if suffix.startswith('identity'):
                    for s_to in pomdp.states:
                        for obs in pomdp.observations:
                            pomdp.observation_probs[action][s_to][obs] = 1.0 if s_to == obs else 0.0
                    i += 1
                    continue
                elif suffix.startswith('uniform'):
                    nO = len(pomdp.observations)
                    for s_to in pomdp.states:
                        for obs in pomdp.observations:
                            pomdp.observation_probs[action][s_to][obs] = 1.0 / nO
                    i += 1
                    continue
                else:
                    # parse matrix
                    i += 1
                    for s_index, s_to in enumerate(pomdp.states):
                        prob_line = lines[i].split()
                        i += 1
                        for o_index, obs in enumerate(pomdp.observations):
                            pomdp.observation_probs[action][s_to][obs] = float(prob_line[o_index])
                    continue

            # --- Reward lines ---
            rm = reward_pattern.match(line)
            if rm:
                action = rm.group(1).strip()
                s_from = rm.group(2).strip()
                s_to = rm.group(3).strip()
                obs = rm.group(4).strip()
                val = float(rm.group(5).strip())

                possible_s_from = [s_from] if s_from != "*" else pomdp.states
                possible_s_to = [s_to] if s_to != "*" else pomdp.states
                possible_obs = [obs] if obs != "*" else pomdp.observations

                for sf in possible_s_from:
                    for st in possible_s_to:
                        for ob in possible_obs:
                            pomdp.rewards[action][sf][st][ob] = val

                i += 1
                continue

            i += 1

        # 3) Build numeric index lookups (state, action, obs)
        pomdp.state_name_to_id = {s_name: i for i, s_name in enumerate(pomdp.states)}
        pomdp.state_id_to_name = {i: s_name for i, s_name in enumerate(pomdp.states)}

        pomdp.action_name_to_id = {a_name: i for i, a_name in enumerate(pomdp.actions)}
        pomdp.action_id_to_name = {i: a_name for i, a_name in enumerate(pomdp.actions)}

        pomdp.obs_name_to_id = {o_name: i for i, o_name in enumerate(pomdp.observations)}
        pomdp.obs_id_to_name = {i: o_name for i, o_name in enumerate(pomdp.observations)}


        # 4) Create X, O as the lists of numeric indices
        pomdp.X = list(range(len(pomdp.states)))  # e.g., [0,1,2,...]
        pomdp.O = list(range(len(pomdp.observations)))  # e.g., [0,1,2,...]
        pomdp.A = list(range(len(pomdp.actions)))

        # 5) Construct T, Z, R as nested lists
        numA = len(pomdp.actions)
        numS = len(pomdp.states)
        numO = len(pomdp.observations)

        # T[a][s][s']
        pomdp.T = [
            [
                [0.0 for _ in range(numS)]
                for _ in range(numS)
            ]
            for _ in range(numA)
        ]

        # Z[a][s'][o]
        pomdp.Z = [
            [
                [0.0 for _ in range(numO)]
                for _ in range(numS)
            ]
            for _ in range(numA)
        ]

        # R[a][s][s'][o]
        pomdp.R = [
            [
                [
                    [0.0 for _ in range(numO)]
                    for _ in range(numS)
                ]
                for _ in range(numS)
            ]
            for _ in range(numA)
        ]

        # Fill T, Z, R using the dictionary-based data
        for a_name in pomdp.actions:
            a = pomdp.action_name_to_id[a_name]
            for s_from in pomdp.states:
                sf = pomdp.state_name_to_id[s_from]
                for s_to in pomdp.states:
                    st = pomdp.state_name_to_id[s_to]
                    pomdp.T[a][sf][st] = pomdp.transition_probs[a_name][s_from][s_to]
                    for o_name in pomdp.observations:
                        o = pomdp.obs_name_to_id[o_name]
                        pomdp.Z[a][st][o] = pomdp.observation_probs[a_name][s_to][o_name]
                        pomdp.R[a][sf][st][o] = pomdp.rewards[a_name][s_from][s_to][o_name]

        return pomdp


def main(file_path):
    pomdp = POMDPParser.parse_pomdp(file_path)

    # Print some info
    print("Parsed POMDP specification:")
    print(f"  Discount:     {pomdp.discount}")
    print(f"  Values:       {pomdp.values}")
    print(f"  States:       {pomdp.states}")
    print(f"  Actions:      {pomdp.actions}")
    print(f"  Observations: {pomdp.observations}")

    print("\nIndex Lookups:")
    print("  state_name_to_id:", pomdp.state_name_to_id)
    print("  state_id_to_name:", pomdp.state_id_to_name)
    print("  action_name_to_id:", pomdp.action_name_to_id)
    print("  action_id_to_name:", pomdp.action_id_to_name)
    print("  obs_name_to_id:", pomdp.obs_name_to_id)
    print("  obs_id_to_name:", pomdp.obs_id_to_name)

    print("\nX (state indices):", pomdp.X)
    print("O (observation indices):", pomdp.O)

    # Example printing of a few elements in T, Z, R
    print("\nSample from T[a][s][s'] (transition):")
    for a in range(len(pomdp.actions)):
        for s in range(len(pomdp.states)):
            for s2 in range(len(pomdp.states)):
                val = pomdp.T[a][s][s2]
                if val > 0:
                    print(f"T[{a}][{s}][{s2}] = {val}")

    print("\nSample from Z[a][s'][o] (observation):")
    for a in range(len(pomdp.actions)):
        for s2 in range(len(pomdp.states)):
            for o in range(len(pomdp.observations)):
                val = pomdp.Z[a][s2][o]
                if val > 0:
                    print(f"Z[{a}][{s2}][{o}] = {val}")

    print("\nSample from R[a][s][s'][o] (rewards):")
    for a in range(len(pomdp.actions)):
        for s in range(len(pomdp.states)):
            for s2 in range(len(pomdp.states)):
                for o in range(len(pomdp.observations)):
                    val = pomdp.R[a][s][s2][o]
                    if val != 0:
                        print(f"R[{a}][{s}][{s2}][{o}] = {val}")


if __name__ == "__main__":
    main(file_path="tiger.pomdp")
