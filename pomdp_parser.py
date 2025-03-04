#!/usr/bin/env python3
"""
parse_pomdp.py

An extended parser for .POMDP files that:
  - Handles numeric counts for states, actions, observations (e.g. "states: 3201").
    It then creates state labels "0", "1", ..., "3200".
  - Parses 'start:' lines to capture the start distribution for all states.
  - Handles single-line transitions of the form: T: a : s_from : s_to prob
  - Handles single-line observations of the form: O: a : s_to : obs prob
  - Handles multi-line or shorthand transitions/observations ("identity", "uniform"),
    though that may be less likely in numeric-labeled domains.
  - Allows R: lines with wildcards (e.g. R: a : s_from : s_to : obs reward).
  - Builds 0-indexed tensors T, Z, and R (nested lists).
  - Provides X (state indices [0..N-1]), A (action indices [0..A-1]),
    O (observation indices [0..O-1]), plus lookup dicts for each.

Usage:
  python parse_pomdp.py file.pomdp
"""

import re


class POMDPParser:
    """
    A container for POMDP specifications and derived data:
      - discount: float
      - values: typically 'reward'
      - states: list of state labels (strings, e.g. ["0","1",..., "3200"])
      - actions: list of action labels
      - observations: list of observation labels
      - start_dist: list of floats (start distribution over states; same order as states)
      - transition_probs[action][s_from][s_to]
      - observation_probs[action][s_to][obs]
      - rewards[action][s_from][s_to][obs]

    Additionally, it holds:
      - state_name_to_id, state_id_to_name
      - action_name_to_id, action_id_to_name
      - obs_name_to_id,    obs_id_to_name

      - X: list of state indices (0..N-1)
      - A: list of action indices (0..numActions-1)
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

        # Start distribution (parallel to self.states)
        self.start_dist = None

        # Dict-based probabilities and rewards
        self.transition_probs = {}  # transition_probs[a_label][s_from_label][s_to_label]
        self.observation_probs = {}  # observation_probs[a_label][s_to_label][obs_label]
        self.rewards = {}  # rewards[a_label][s_from_label][s_to_label][obs_label]

        # Index-based lookups
        self.state_name_to_id = {}
        self.state_id_to_name = {}
        self.action_name_to_id = {}
        self.action_id_to_name = {}
        self.obs_name_to_id = {}
        self.obs_id_to_name = {}

        # Numeric sets (lists) of indices
        self.X = []  # states
        self.A = []  # actions
        self.O = []  # observations

        # Tensor-like nested lists
        self.T = []  # T[a][s][s']
        self.Z = []  # Z[a][s'][o]
        self.R = []  # R[a][s][s'][o]

    def initialize_transitions(self):
        for a in self.actions:
            print(f"initializing transition {a}/{len(self.actions) - 1}")
            self.transition_probs[a] = {}
            for s_from in self.states:
                self.transition_probs[a][s_from] = {}
                for s_to in self.states:
                    self.transition_probs[a][s_from][s_to] = 0.0

    def initialize_observations(self):
        for a in self.actions:
            print(f"initializing observation {a}/{len(self.actions) - 1}")
            self.observation_probs[a] = {}
            for s_to in self.states:
                self.observation_probs[a][s_to] = {}
                for obs in self.observations:
                    self.observation_probs[a][s_to][obs] = 0.0

    def initialize_rewards(self):
        for a in self.actions:
            print(f"initializing reward {a}/{len(self.actions) - 1}")
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
          - discount, values, states, actions, observations, start_dist
          - transition_probs, observation_probs, rewards (dict-of-dicts)
          - T, Z, R as nested Python lists (indexed from 0)
          - X, A, O as lists of numeric indices
          - Lookup dicts (state_name_to_id, etc.)
        """
        pomdp = POMDPParser()

        with open(file_path, 'r') as f:
            # Keep non-empty, non-comment lines
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

        # Patterns for basic fields
        discount_pattern = re.compile(r"^discount\s*:\s*([\d\.]+)")
        values_pattern = re.compile(r"^values\s*:\s*(\S+)")

        # If the line after "states:" is a single integer, interpret that as the number of states
        # Else interpret it as space-separated state labels.
        states_pattern = re.compile(r"^states\s*:\s*(.*)")
        actions_pattern = re.compile(r"^actions\s*:\s*(.*)")
        obs_pattern = re.compile(r"^observations\s*:\s*(.*)")
        start_pattern = re.compile(r"^start\s*:\s*(.*)")

        # Single-line T: a : s_from : s_to prob
        transition_spec_pattern = re.compile(
            r"^T\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s+([\-\d\.]+)\s*$"
        )
        # Single-line O: a : s_to : obs prob
        observation_spec_pattern = re.compile(
            r"^O\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s+([\-\d\.]+)\s*$"
        )

        # T/O lines (e.g. T: a, might be identity/uniform or a matrix)
        transition_pattern = re.compile(r"^T\s*:\s*([^:]+)(.*)")
        observation_pattern = re.compile(r"^O\s*:\s*([^:]+)(.*)")

        # Reward lines with possible wildcards
        reward_pattern = re.compile(
            r"^R\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s*:\s*([^:]+)\s+([-]?\d+(\.\d+)?)"
        )

        # 1) Parse basic fields (discount, values, states, actions, observations, start)
        i = 0
        while i < len(lines):
            print(f"Parsing line {i+1}/{len(lines)} [First pass]")
            line = lines[i]

            m_discount = discount_pattern.match(line)
            if m_discount:
                pomdp.discount = float(m_discount.group(1))
                i += 1
                continue

            m_values = values_pattern.match(line)
            if m_values:
                pomdp.values = m_values.group(1)
                i += 1
                continue

            m_states = states_pattern.match(line)
            if m_states:
                possible_str = m_states.group(1).strip()  # e.g. "3201" or "s0 s1 s2"
                # Attempt to interpret as integer
                if re.fullmatch(r"\d+", possible_str):
                    # e.g. "3201" => create labels "0", "1", ..., "3200"
                    count_states = int(possible_str)
                    pomdp.states = list(map(str, range(count_states)))
                else:
                    # Otherwise interpret as space-separated labels
                    pomdp.states = possible_str.split()
                i += 1
                continue

            m_actions = actions_pattern.match(line)
            if m_actions:
                possible_str = m_actions.group(1).strip()  # e.g. "12" or "listen open-left open-right"
                if re.fullmatch(r"\d+", possible_str):
                    count_actions = int(possible_str)
                    pomdp.actions = list(map(str, range(count_actions)))
                else:
                    pomdp.actions = possible_str.split()
                i += 1
                continue

            m_obs = obs_pattern.match(line)
            if m_obs:
                possible_str = m_obs.group(1).strip()  # e.g. "2" or "ogood obad"
                if re.fullmatch(r"\d+", possible_str):
                    count_observations = int(possible_str)
                    pomdp.observations = list(map(str, range(count_observations)))
                else:
                    pomdp.observations = possible_str.split()
                i += 1
                continue

            m_start = start_pattern.match(line)
            if m_start:
                # parse the entire line after 'start:'
                start_str = m_start.group(1).strip()
                pomdp.start_dist = [float(x) for x in start_str.split()]
                i += 1
                continue

            i += 1

        # 2) Initialize data structures now that we know states/actions/obs
        print("Initializing data structures...")
        pomdp.initialize_transitions()
        pomdp.initialize_observations()
        pomdp.initialize_rewards()

        # 3) Second pass: parse T, O, R lines
        i = 0
        while i < len(lines):
            print(f"Parsing line {i+1}/{len(lines)} [Second pass]")
            line = lines[i]

            # Single-line transitions
            tm_single = transition_spec_pattern.match(line)
            if tm_single:
                action = tm_single.group(1).strip()
                s_from = tm_single.group(2).strip()
                s_to = tm_single.group(3).strip()
                prob = float(tm_single.group(4))

                # Ensure these are known
                if action not in pomdp.actions:
                    print(f"Warning: action '{action}' not in declared actions.")
                if s_from not in pomdp.states:
                    print(f"Warning: state_from '{s_from}' not in declared states.")
                if s_to not in pomdp.states:
                    print(f"Warning: state_to '{s_to}' not in declared states.")

                # Safely set (if valid)
                if action in pomdp.actions and s_from in pomdp.states and s_to in pomdp.states:
                    pomdp.transition_probs[action][s_from][s_to] = prob

                i += 1
                continue

            # Single-line observations
            om_single = observation_spec_pattern.match(line)
            if om_single:
                action = om_single.group(1).strip()
                s_to = om_single.group(2).strip()
                obs = om_single.group(3).strip()
                prob = float(om_single.group(4))

                if action not in pomdp.actions:
                    print(f"Warning: action '{action}' not in declared actions.")
                if s_to not in pomdp.states:
                    print(f"Warning: state_to '{s_to}' not in declared states.")
                if obs not in pomdp.observations:
                    print(f"Warning: obs '{obs}' not in declared observations.")

                if action in pomdp.actions and s_to in pomdp.states and obs in pomdp.observations:
                    pomdp.observation_probs[action][s_to][obs] = prob

                i += 1
                continue

            # T: action with possible identity/uniform or matrix
            tm = transition_pattern.match(line)
            if tm:
                action = tm.group(1).strip()
                suffix = tm.group(2).strip()

                if action not in pomdp.actions:
                    print(f"Warning: action '{action}' not in declared actions.")

                if not suffix:
                    # maybe next line is identity/uniform/matrix
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('identity'):
                            for sf in pomdp.states:
                                for st in pomdp.states:
                                    pomdp.transition_probs[action][sf][st] = (
                                        1.0 if sf == st else 0.0
                                    )
                            i += 2
                            continue
                        elif next_line.startswith('uniform'):
                            nS = len(pomdp.states)
                            for sf in pomdp.states:
                                for st in pomdp.states:
                                    pomdp.transition_probs[action][sf][st] = 1.0 / nS
                            i += 2
                            continue
                    # Otherwise we attempt to read a matrix
                    i += 1
                    for s_index, sf in enumerate(pomdp.states):
                        if i >= len(lines):
                            break
                        row_candidate = lines[i]
                        # If we see something like "O: ..." or "T: ..." or "R: ...", break
                        if re.match(r"^[TOR]:", row_candidate):
                            break
                        row_parts = row_candidate.split()
                        # If row_parts length != number_of_states, might be invalid or new section
                        if len(row_parts) != len(pomdp.states):
                            break
                        # parse as floats
                        for st_index, st in enumerate(pomdp.states):
                            val = float(row_parts[st_index])
                            pomdp.transition_probs[action][sf][st] = val
                        i += 1
                    continue

                # if suffix is on the same line
                if suffix.startswith('identity'):
                    for sf in pomdp.states:
                        for st in pomdp.states:
                            pomdp.transition_probs[action][sf][st] = (1.0 if sf == st else 0.0)
                    i += 1
                    continue
                elif suffix.startswith('uniform'):
                    nS = len(pomdp.states)
                    for sf in pomdp.states:
                        for st in pomdp.states:
                            pomdp.transition_probs[action][sf][st] = 1.0 / nS
                    i += 1
                    continue
                else:
                    # parse matrix lines
                    i += 1
                    for s_index, sf in enumerate(pomdp.states):
                        if i >= len(lines):
                            break
                        row_candidate = lines[i]
                        if re.match(r"^[TOR]:", row_candidate):
                            break
                        row_parts = row_candidate.split()
                        if len(row_parts) != len(pomdp.states):
                            break
                        for st_index, st in enumerate(pomdp.states):
                            val = float(row_parts[st_index])
                            pomdp.transition_probs[action][sf][st] = val
                        i += 1
                    continue

            # O: action with possible identity/uniform or matrix
            om = observation_pattern.match(line)
            if om:
                action = om.group(1).strip()
                suffix = om.group(2).strip()

                if action not in pomdp.actions:
                    print(f"Warning: action '{action}' not in declared actions.")

                if not suffix:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('identity'):
                            for st in pomdp.states:
                                for ob in pomdp.observations:
                                    pomdp.observation_probs[action][st][ob] = (
                                        1.0 if st == ob else 0.0
                                    )
                            i += 2
                            continue
                        elif next_line.startswith('uniform'):
                            nO = len(pomdp.observations)
                            for st in pomdp.states:
                                for ob in pomdp.observations:
                                    pomdp.observation_probs[action][st][ob] = 1.0 / nO
                            i += 2
                            continue
                    # parse matrix
                    i += 1
                    for s_index, st in enumerate(pomdp.states):
                        if i >= len(lines):
                            break
                        row_candidate = lines[i]
                        if re.match(r"^[TOR]:", row_candidate):
                            break
                        row_parts = row_candidate.split()
                        if len(row_parts) != len(pomdp.observations):
                            break
                        for o_index, ob in enumerate(pomdp.observations):
                            pomdp.observation_probs[action][st][ob] = float(row_parts[o_index])
                        i += 1
                    continue

                # suffix on same line
                if suffix.startswith('identity'):
                    for st in pomdp.states:
                        for ob in pomdp.observations:
                            pomdp.observation_probs[action][st][ob] = (1.0 if st == ob else 0.0)
                    i += 1
                    continue
                elif suffix.startswith('uniform'):
                    nO = len(pomdp.observations)
                    for st in pomdp.states:
                        for ob in pomdp.observations:
                            pomdp.observation_probs[action][st][ob] = 1.0 / nO
                    i += 1
                    continue
                else:
                    # parse matrix
                    i += 1
                    for s_index, st in enumerate(pomdp.states):
                        if i >= len(lines):
                            break
                        row_candidate = lines[i]
                        if re.match(r"^[TOR]:", row_candidate):
                            break
                        row_parts = row_candidate.split()
                        if len(row_parts) != len(pomdp.observations):
                            break
                        for o_index, ob in enumerate(pomdp.observations):
                            pomdp.observation_probs[action][st][ob] = float(row_parts[o_index])
                        i += 1
                    continue

            # Reward lines
            rm = reward_pattern.match(line)
            if rm:
                a_label = rm.group(1).strip()
                s_from = rm.group(2).strip()
                s_to = rm.group(3).strip()
                obs = rm.group(4).strip()
                val = float(rm.group(5).strip())

                possible_s_from = [s_from] if s_from != "*" else pomdp.states
                possible_s_to = [s_to] if s_to != "*" else pomdp.states
                possible_obs = [obs] if obs != "*" else pomdp.observations

                if a_label not in pomdp.actions:
                    print(f"Warning: action '{a_label}' not recognized.")

                for sf in possible_s_from:
                    if sf not in pomdp.states:
                        print(f"Warning: state_from '{sf}' not recognized.")
                        continue
                    for st in possible_s_to:
                        if st not in pomdp.states:
                            print(f"Warning: state_to '{st}' not recognized.")
                            continue
                        for ob in possible_obs:
                            if ob not in pomdp.observations:
                                print(f"Warning: obs '{ob}' not recognized.")
                                continue
                            if a_label in pomdp.actions:
                                pomdp.rewards[a_label][sf][st][ob] = val

                i += 1
                continue

            i += 1

        # 4) Build numeric index mappings
        print("Building numeric index mappings...")
        pomdp.state_name_to_id = {nm: idx for idx, nm in enumerate(pomdp.states)}
        pomdp.state_id_to_name = {idx: nm for idx, nm in enumerate(pomdp.states)}

        pomdp.action_name_to_id = {nm: idx for idx, nm in enumerate(pomdp.actions)}
        pomdp.action_id_to_name = {idx: nm for idx, nm in enumerate(pomdp.actions)}

        pomdp.obs_name_to_id = {nm: idx for idx, nm in enumerate(pomdp.observations)}
        pomdp.obs_id_to_name = {idx: nm for idx, nm in enumerate(pomdp.observations)}

        # 5) Create X, A, O
        print("Creating X, A, O...")
        pomdp.X = list(range(len(pomdp.states)))
        pomdp.A = list(range(len(pomdp.actions)))
        pomdp.O = list(range(len(pomdp.observations)))

        # 6) Construct T, Z, R as nested lists
        print("Constructing T, Z, R...")
        numA = len(pomdp.actions)
        numS = len(pomdp.states)
        numO = len(pomdp.observations)
        print("Constructing T")
        pomdp.T = []
        for a in range(numA):
            print(f"Processing action {a + 1}/{numA}")  # Print progress
            action_matrix = []
            for s in range(numS):
                print(f"Processing state {s + 1}/{numS}")  # Print progress
                state_transitions = [0.0 for _ in range(numS)]
                action_matrix.append(state_transitions)
            pomdp.T.append(action_matrix)

        print("Constructing Z")
        pomdp.Z = []
        for a in range(numA):
            print(f"Processing action {a + 1}/{numA} for Z")
            action_observation_matrix = []
            for s in range(numS):
                print(f"Processing state {s + 1}/{numS}")  # Print progress
                state_observations = [0.0 for _ in range(numO)]
                action_observation_matrix.append(state_observations)
            pomdp.Z.append(action_observation_matrix)

        print("Constructing R")
        pomdp.R = []
        for a in range(numA):
            print(f"Processing action {a + 1}/{numA} for R")
            action_reward_matrix = []
            for s1 in range(numS):
                print(f"Processing state {s1 + 1}/{numS}")
                state1_rewards = []
                for s2 in range(numS):
                    print(f"Processing state {s2 + 1}/{numS}")
                    observation_rewards = [0.0 for _ in range(numO)]
                    state1_rewards.append(observation_rewards)
                action_reward_matrix.append(state1_rewards)
            pomdp.R.append(action_reward_matrix)

        # Fill T, Z, R
        print("Filling T, Z, R...")
        for i, a_label in enumerate(pomdp.actions):
            print(f"Filling {i}/{len(pomdp.actions)}")
            a_id = pomdp.action_name_to_id[a_label]
            for sf in pomdp.states:
                sf_id = pomdp.state_name_to_id[sf]
                for st in pomdp.states:
                    st_id = pomdp.state_name_to_id[st]
                    pomdp.T[a_id][sf_id][st_id] = pomdp.transition_probs[a_label][sf][st]
                    for ob in pomdp.observations:
                        ob_id = pomdp.obs_name_to_id[ob]
                        pomdp.Z[a_id][st_id][ob_id] = pomdp.observation_probs[a_label][st][ob]
                        pomdp.R[a_id][sf_id][st_id][ob_id] = pomdp.rewards[a_label][sf][st][ob]
        return pomdp


def main(file_path):
    pomdp = POMDPParser.parse_pomdp(file_path)

    print("Parsed POMDP specification:\n")
    print(f"Discount:       {pomdp.discount}")
    print(f"Values:         {pomdp.values}")
    print(f"States:         {pomdp.states[:20]} ... (total {len(pomdp.states)})")
    print(f"Actions:        {pomdp.actions}")
    print(f"Observations:   {pomdp.observations}")
    if pomdp.start_dist is not None:
        print(f"Start Dist (first 20): {pomdp.start_dist[:20]} ... (total {len(pomdp.start_dist)})")

    print("\nIndex Lookups:")
    print("  state_name_to_id (sample):", {k: pomdp.state_name_to_id[k]
                                           for k in list(pomdp.state_name_to_id)[:5]})
    print("  action_name_to_id:", pomdp.action_name_to_id)
    print("  obs_name_to_id:", pomdp.obs_name_to_id)

    print("\nX (state indices) length:", len(pomdp.X))
    print("A (action indices):", pomdp.A)
    print("O (observation indices):", pomdp.O)

    print("\nCheck a few transitions:")
    for a in range(len(pomdp.actions)):
        for s in range(len(pomdp.states)):
            for s2 in range(len(pomdp.states)):
                val = pomdp.T[a][s][s2]
                if val != 0.0:
                    print(f"T[{a}][{s}][{s2}] = {val}")
                    # limit printing
                    break
            break


if __name__ == "__main__":
    main("RockSample_5_7.pomdp")
