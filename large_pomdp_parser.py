import pickle
import numpy as np

class SparseRowT:
    """
    Sparse or lazy representation for transitions: T[a][s][s_next].
    - default_type in {None, 'identity', 'uniform'}
    - explicit is a dict { s_from -> ROW }, where ROW can be:
        (a) dict { s_to -> prob } for sparse,
        (b) list[float] for dense,
        (c) ("identity",) or ("uniform",) for special row,
        (d) None => fallback to the action-level default_type.
    """
    __slots__ = ("num_states", "default_type", "explicit")

    def __init__(self, num_states):
        self.num_states = num_states
        self.default_type = None   # or 'identity', 'uniform'
        self.explicit = {}         # s_from -> rowSpec

    def set_entire_identity(self):
        """Mark the entire T for this action as identity for all s_from."""
        self.default_type = "identity"

    def set_entire_uniform(self):
        """Mark the entire T for this action as uniform for all s_from."""
        self.default_type = "uniform"

    def set_sparse_entries(self, s_from, s_to_dict):
        """
        Store a row as a dict { s_to -> prob }, replacing any existing row for s_from.
        """
        self.explicit[s_from] = s_to_dict

    def set_distribution(self, s_from, dist_list):
        """
        Store a row as a dense list of floats (length = num_states).
        """
        self.explicit[s_from] = dist_list

    def set_row_identity(self, s_from):
        self.explicit[s_from] = ("identity",)

    def set_row_uniform(self, s_from):
        self.explicit[s_from] = ("uniform",)

    def get_prob(self, s_from, s_to):
        """Return T[a][s_from][s_to]."""
        row = self.explicit.get(s_from)
        if row is None:
            # fallback to default
            if self.default_type == "identity":
                return 1.0 if (s_to == s_from) else 0.0
            elif self.default_type == "uniform":
                return 1.0 / self.num_states
            else:
                return 0.0

        if isinstance(row, dict):
            return row.get(s_to, 0.0)
        elif isinstance(row, list):
            return row[s_to]
        elif isinstance(row, tuple):
            # e.g. ("identity",)
            mode = row[0]
            if mode == "identity":
                return 1.0 if (s_to == s_from) else 0.0
            elif mode == "uniform":
                return 1.0 / self.num_states
            else:
                return 0.0
        else:
            return 0.0

    def get_successor_states(self, s_from):
        """
        Return a list of (s_next, probability) for all s_next with prob>0 under this action.
        """
        row = self.explicit.get(s_from)
        if row is None:
            # fallback to default
            if self.default_type == "identity":
                return [(s_from, 1.0)]
            elif self.default_type == "uniform":
                return [(s_to, 1.0/self.num_states) for s_to in range(self.num_states)]
            else:
                return []
        if isinstance(row, dict):
            return [(s_to, p) for (s_to, p) in row.items() if p != 0.0]
        elif isinstance(row, list):
            return [(s_to, p) for (s_to, p) in enumerate(row) if p != 0.0]
        elif isinstance(row, tuple):
            mode = row[0]
            if mode == "identity":
                return [(s_from, 1.0)]
            elif mode == "uniform":
                return [(s_to, 1.0/self.num_states) for s_to in range(self.num_states)]
            else:
                return []
        else:
            return []


class SparseRowO:
    """
    Sparse or lazy representation for observations: O[a][s_next][obs].
    - default_type in {None, 'identity', 'uniform'}
    - explicit[s_next] -> rowSpec, which can be:
       dict{ obs->p }, list[float], or ("identity",)/("uniform",).
    """
    __slots__ = ("num_states", "num_obs", "default_type", "explicit")

    def __init__(self, num_states, num_obs):
        self.num_states = num_states
        self.num_obs = num_obs
        self.default_type = None
        self.explicit = {}

    def set_entire_identity(self):
        self.default_type = "identity"

    def set_entire_uniform(self):
        self.default_type = "uniform"

    def set_sparse_entries(self, s_next, obs_dict):
        self.explicit[s_next] = obs_dict

    def set_distribution(self, s_next, dist_list):
        self.explicit[s_next] = dist_list

    def get_prob(self, s_next, obs):
        """Return O[a][s_next][obs]."""
        row = self.explicit.get(s_next)
        if row is None:
            # fallback to default
            if self.default_type == "identity":
                # Only if obs == s_next and num_obs == num_states
                if self.num_obs == self.num_states:
                    return 1.0 if (obs == s_next) else 0.0
                else:
                    return 0.0
            elif self.default_type == "uniform":
                return 1.0 / self.num_obs
            else:
                return 0.0

        if isinstance(row, dict):
            return row.get(obs, 0.0)
        elif isinstance(row, list):
            return row[obs]
        elif isinstance(row, tuple):
            mode = row[0]
            if mode == "identity":
                if self.num_obs == self.num_states:
                    return 1.0 if (obs == s_next) else 0.0
                else:
                    return 0.0
            elif mode == "uniform":
                return 1.0 / self.num_obs
            else:
                return 0.0
        else:
            return 0.0

    def get_nonzero_observations(self, s_next):
        """
        Return a list of (obs_id, prob) for all obs with prob>0 given s_next, action a.
        """
        row = self.explicit.get(s_next)
        if row is None:
            # fallback
            if self.default_type == "identity":
                if self.num_obs == self.num_states:
                    return [(s_next, 1.0)]
                else:
                    return []
            elif self.default_type == "uniform":
                return [(o, 1.0/self.num_obs) for o in range(self.num_obs)]
            else:
                return []
        if isinstance(row, dict):
            return [(o, p) for (o, p) in row.items() if p != 0.0]
        elif isinstance(row, list):
            return [(o, p) for (o, p) in enumerate(row) if p != 0.0]
        elif isinstance(row, tuple):
            mode = row[0]
            if mode == "identity":
                if self.num_obs == self.num_states:
                    return [(s_next, 1.0)]
                else:
                    return []
            elif mode == "uniform":
                return [(o, 1.0/self.num_obs) for o in range(self.num_obs)]
            else:
                return []
        else:
            return []


################################################################################
# 2. The POMDP Parser (pure data, no local closures)
################################################################################

def parse_pomdp_data(filename):
    """
    Parse a .pomdp file into a dictionary-based model with numeric IDs for states/actions/obs,
    and a memory-efficient representation for T/O.

    The returned 'model' is purely data (no nested local functions), so it's picklable.
    """

    import sys

    discount = None
    values = None

    states_list = []
    actions_list = []
    obs_list = []

    state_index = {}
    action_index = {}
    obs_index = {}

    num_states = None
    num_actions = None
    num_obs = None

    start_dist = []
    R_dict = {}   # (a_id, s_id, s_next, o_id) -> reward

    T_structs = []  # T[a_id] = SparseRowT(num_states)
    O_structs = []  # Z[a_id] = SparseRowO(num_states, num_obs)

    parse_mode = None     # 'T-matrix' or 'O-matrix'
    current_action = None
    matrix_lines = []

    def strip_comment(line):
        if '#' in line:
            line = line[: line.index('#')]
        return line.strip()

    def split_colon(line):
        return [x.strip() for x in line.split(':')]

    def ensure_T_and_O_exist():
        """Allocate T_structs and O_structs once #states, #actions, #obs are known."""
        if not T_structs and (num_actions is not None) and (num_states is not None) and (num_obs is not None):
            for _ in range(num_actions):
                T_structs.append(SparseRowT(num_states))
                O_structs.append(SparseRowO(num_states, num_obs))

    with open(filename, 'r') as f:
        line_idx = 0
        for raw_line in f:
            print(f"processing line {line_idx}")
            line_idx += 1
            line = strip_comment(raw_line)
            if not line:
                continue

            if parse_mode == 'T-matrix':
                line_lower = line.lower()
                if line_lower == "identity":
                    a_id = action_index[current_action]
                    T_structs[a_id].set_entire_identity()
                    parse_mode = None
                    current_action = None
                    matrix_lines = []
                    continue
                elif line_lower == "uniform":
                    a_id = action_index[current_action]
                    T_structs[a_id].set_entire_uniform()
                    parse_mode = None
                    current_action = None
                    matrix_lines = []
                    continue
                else:
                    matrix_lines.append(line)
                    if len(matrix_lines) == num_states:
                        # finalize T
                        a_id = action_index[current_action]
                        for s_id in range(num_states):
                            row_strs = matrix_lines[s_id].split()
                            if len(row_strs) != num_states:
                                raise ValueError("Transition matrix row length mismatch.")
                            threshold = min(50, num_states // 10)
                            nonzero = []
                            for s_next, val_str in enumerate(row_strs):
                                if val_str not in ('0','0.0','0.000000'):
                                    p = float(val_str)
                                    if p != 0.0:
                                        nonzero.append((s_next, p))
                            if len(nonzero) < threshold:
                                row_dict = {st: pr for (st, pr) in nonzero}
                                T_structs[a_id].set_sparse_entries(s_id, row_dict)
                            else:
                                row_list = [float(x) for x in row_strs]
                                T_structs[a_id].set_distribution(s_id, row_list)

                        parse_mode = None
                        current_action = None
                        matrix_lines = []
                continue

            elif parse_mode == 'O-matrix':
                line_lower = line.lower()
                if line_lower == "identity":
                    a_id = action_index[current_action]
                    O_structs[a_id].set_entire_identity()
                    parse_mode = None
                    current_action = None
                    matrix_lines = []
                    continue
                elif line_lower == "uniform":
                    a_id = action_index[current_action]
                    O_structs[a_id].set_entire_uniform()
                    parse_mode = None
                    current_action = None
                    matrix_lines = []
                    continue
                else:
                    matrix_lines.append(line)
                    if len(matrix_lines) == num_states:
                        # finalize O
                        a_id = action_index[current_action]
                        for s_next in range(num_states):
                            row_strs = matrix_lines[s_next].split()
                            if len(row_strs) != num_obs:
                                raise ValueError("Observation matrix row length mismatch.")
                            threshold = min(50, num_obs // 10)
                            nonzero = []
                            for o_id, val_str in enumerate(row_strs):
                                if val_str not in ('0','0.0','0.000000'):
                                    p = float(val_str)
                                    if p != 0.0:
                                        nonzero.append((o_id, p))
                            if len(nonzero) < threshold:
                                row_dict = {ob: pr for (ob,pr) in nonzero}
                                O_structs[a_id].set_sparse_entries(s_next, row_dict)
                            else:
                                row_list = [float(x) for x in row_strs]
                                O_structs[a_id].set_distribution(s_next, row_list)

                        parse_mode = None
                        current_action = None
                        matrix_lines = []
                continue

            # Otherwise parse normal lines
            lower_line = line.lower()
            if lower_line.startswith("discount:"):
                parts = split_colon(line)
                discount = float(parts[1])
            elif lower_line.startswith("values:"):
                parts = split_colon(line)
                values = parts[1].strip()
            elif lower_line.startswith("states:"):
                parts = split_colon(line)
                tokens = parts[1].split()
                if len(tokens) == 1:
                    try:
                        num_states = int(tokens[0])
                        states_list = [str(i) for i in range(num_states)]
                    except ValueError:
                        states_list = [tokens[0]]
                        num_states = 1
                else:
                    states_list = tokens
                    num_states = len(tokens)
                for i_s, s_name in enumerate(states_list):
                    state_index[s_name] = i_s

            elif lower_line.startswith("actions:"):
                parts = split_colon(line)
                tokens = parts[1].split()
                if len(tokens) == 1:
                    try:
                        num_actions = int(tokens[0])
                        actions_list = [str(i) for i in range(num_actions)]
                    except ValueError:
                        actions_list = [tokens[0]]
                        num_actions = 1
                else:
                    actions_list = tokens
                    num_actions = len(tokens)
                for i_a, a_name in enumerate(actions_list):
                    action_index[a_name] = i_a

            elif lower_line.startswith("observations:"):
                parts = split_colon(line)
                tokens = parts[1].split()
                if len(tokens) == 1:
                    try:
                        num_obs = int(tokens[0])
                        obs_list = [str(i) for i in range(num_obs)]
                    except ValueError:
                        obs_list = [tokens[0]]
                        num_obs = 1
                else:
                    obs_list = tokens
                    num_obs = len(tokens)
                for i_o, o_name in enumerate(obs_list):
                    obs_index[o_name] = i_o

            elif lower_line.startswith("start:"):
                parts = split_colon(line)
                tokens = parts[1].split()
                start_dist = [float(x) for x in tokens]

            elif lower_line.startswith("t:"):
                # parse transitions
                ensure_T_and_O_exist()
                parts = split_colon(line)
                parts = parts[1:]  # remove 'T'
                if not parts:
                    raise ValueError("No action in T line.")
                a_str = parts[0]
                a_id = action_index[a_str]
                if len(parts) == 1:
                    # means next lines => identity/uniform or matrix
                    parse_mode = 'T-matrix'
                    current_action = a_str
                elif len(parts) == 2:
                    sub = parts[1].lower()
                    if sub == 'identity':
                        T_structs[a_id].set_entire_identity()
                    elif sub == 'uniform':
                        T_structs[a_id].set_entire_uniform()
                    else:
                        raise ValueError("Unsupported T: a : ??? line.")
                else:
                    # T: a : s : s_next prob  (not fully implemented for brevity)
                    pass

            elif lower_line.startswith("o:"):
                # parse observations
                ensure_T_and_O_exist()
                parts = split_colon(line)
                parts = parts[1:]
                if not parts:
                    raise ValueError("No action in O line.")
                a_str = parts[0]
                a_id = action_index[a_str]
                if len(parts) == 1:
                    parse_mode = 'O-matrix'
                    current_action = a_str
                elif len(parts) == 2:
                    sub = parts[1].lower()
                    if sub == 'identity':
                        O_structs[a_id].set_entire_identity()
                    elif sub == 'uniform':
                        O_structs[a_id].set_entire_uniform()
                    else:
                        raise ValueError("Unsupported O short line.")
                else:
                    # O: a : s_next : obs prob
                    pass

            elif lower_line.startswith("r:"):
                # parse rewards
                ensure_T_and_O_exist()
                parts = split_colon(line)
                parts = parts[1:]
                if len(parts) < 2:
                    raise ValueError("Invalid R line.")
                a_str = parts[0]
                a_id = action_index[a_str]
                s_candidates = range(num_states)
                s_next_candidates = range(num_states)
                obs_candidates = range(num_obs)
                if len(parts) >= 2:
                    s_str = parts[1]
                    if s_str != "*":
                        s_candidates = [state_index[s_str]] if s_str in state_index else [int(s_str)]
                if len(parts) >= 3:
                    s_next_str = parts[2]
                    if s_next_str != "*":
                        s_next_candidates = [state_index[s_next_str]] if s_next_str in state_index else [int(s_next_str)]
                if len(parts) < 4:
                    raise ValueError("R line missing obs:reward segment.")
                obs_and_rew = parts[3].split()
                if len(obs_and_rew) == 2:
                    o_str, rew_str = obs_and_rew
                    rew_val = float(rew_str)
                    if o_str == "*":
                        for sf in s_candidates:
                            for st in s_next_candidates:
                                for ob in obs_candidates:
                                    R_dict[(a_id, sf, st, ob)] = rew_val
                    else:
                        o_id = obs_index[o_str] if o_str in obs_index else int(o_str)
                        for sf in s_candidates:
                            for st in s_next_candidates:
                                R_dict[(a_id, sf, st, o_id)] = rew_val
                else:
                    raise ValueError("Multiple obs:reward pairs on one R line not supported here.")

            else:
                raise ValueError(f"Unrecognized line format: '{line}'")

    # If start distribution is missing, default to uniform
    if not start_dist and num_states is not None:
        start_dist = [1.0 / num_states] * num_states

    # finalize T/O if not allocated
    if not T_structs and (num_actions is not None) and (num_states is not None) and (num_obs is not None):
        for _ in range(num_actions):
            T_structs.append(SparseRowT(num_states))
            O_structs.append(SparseRowO(num_states, num_obs))

    # Build the final model dictionary (pure data, no local function references)
    model = {
        # ID->string
        "states": states_list,
        "actions": actions_list,
        "observations": obs_list,

        # string->ID
        "state_index": state_index,
        "action_index": action_index,
        "obs_index": obs_index,

        "start": start_dist,
        "discount": discount,
        "values": values,

        "T": T_structs,  # T[a_id] = SparseRowT
        "Z": O_structs,  # Z[a_id] = SparseRowO
        "R": R_dict,     # (a_id, s_id, s_next, o_id)->reward
    }
    return model


################################################################################
# 3. Top-level helper functions (not closures) to operate on the model
################################################################################

def get_successor_states(model, s_id, a_id):
    """
    Return a list of (s_next_id, prob) with prob>0 for T[a_id][s_id].
    """
    return model["T"][a_id].get_successor_states(s_id)

def get_next_state_obs_pairs(model, s_id, a_id):
    """
    Return a list of (s_next_id, obs_id, joint_prob),
    where joint_prob = T[a_id][s_id][s_next_id] * Z[a_id][s_next_id][obs_id],
    for all non-zero entries.
    """
    result = []
    succ = model["T"][a_id].get_successor_states(s_id)
    for (sn, pT) in succ:
        if pT == 0.0:
            continue
        obs_list = model["Z"][a_id].get_nonzero_observations(sn)
        for (o, pO) in obs_list:
            p = pT * pO
            if p > 0.0:
                result.append((sn, o, p))
    return result


################################################################################
# 4. Save / Load functions using pickle
################################################################################

def save_model(model, filename):
    """
    Save the entire 'model' (pure data + SparseRowT/O objects) to a file using pickle.
    Because we have no nested local functions in 'model', we can pickle it successfully.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Load a previously saved model from disk with pickle.
    Make sure you have the same SparseRowT / SparseRowO definitions in scope
    before calling this.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def sample_next_state_and_obs(model, s_id, a_id):
    """
    From current state s_id and action a_id, sample:
      - next state s_next_id from T
      - observation obs_id from O
    Returns (s_next_id, obs_id).
    If either distribution is empty, returns (None, None) or (s_next_id, None).
    """
    # 1) Sample next state
    succ = model["T"][a_id].get_successor_states(s_id)
    if not succ:
        # no successor states => empty distribution
        return (None, None)

    states, probs = zip(*succ)
    s_next = np.random.choice(states, p=probs)

    # 2) Sample observation
    obs_candidates = model["Z"][a_id].get_nonzero_observations(s_next)
    if not obs_candidates:
        # no nonzero obs => e.g. no observation
        return (s_next, None)

    obs_ids, obs_probs = zip(*obs_candidates)
    z = np.random.choice(obs_ids, p=obs_probs)

    return (s_next, z)

################################################################################
# 5. Example usage
################################################################################

if __name__ == "__main__":
    # Example usage with the "tiger" .pomdp file
    parsed_model = parse_pomdp_data("tiger.pomdp")

    # Save to disk
    save_model(parsed_model, "tiger_model.pkl")

    # Load back
    loaded_model = load_model("tiger_model.pkl")

    # Suppose we want to get successors from state "tiger-left" under action "listen"
    s_id = loaded_model["state_index"]["tiger-left"]
    a_id = loaded_model["action_index"]["listen"]
    successors = get_successor_states(loaded_model, s_id, a_id)
    print("Successor states for (s='tiger-left', a='listen'):", successors)

    # Or get the (s_next, obs) pairs with joint probability
    sn_obs = get_next_state_obs_pairs(loaded_model, s_id, a_id)
    print("Next-state & obs pairs (s_next, obs, prob):", sn_obs)


# if __name__ == '__main__':
#     model = parse_pomdp_with_joint_successors("tiger.pomdp")
#     save_model(model, "tiger.pkl")
#     # print(model)
#     # print(model["states"])
#     # print(model["actions"])
#     # print(model["observations"])
#     # print(model["state_index"])
#     # print(model["action_index"])
#     # print(model["obs_index"])
#     # print(model["start"])
#     # print(model["discount"])
#     # print(model["values"])
#     # print(model["T"])
#     # print(model["Z"])
#     s_id = model["state_index"][model["states"][0]]
#     a_id = model["action_index"][model["actions"][0]]
#     o_id = model["obs_index"][model["observations"][0]]
#
#     # Then get all next states with nonzero prob:
#     pairs = model["get_next_state_obs_pairs"](s_id, a_id)
#     print(len(pairs), "pairs")
#     # pairs is a list of (s_next_id, obs_id, probability).
#
#     for (sn, o, p) in pairs:
#         print(" s -> s_next,obs => prob = ", s_id, "->", (sn, o), "=", p)
#         s_name = model["states"][s_id]
#         a_name = model["actions"][a_id]
#         s_next_name = model["states"][sn]
#         o_name = model["observations"][o]
#         print(model["R"])
#         r = 0
#         if (a_id, s_id, sn, o) in model["R"]:
#             r = model["R"][(a_id, s_id, sn, o)]
#         print("Reward for", s_name, a_name, s_next_name, o_name, "is", r)
#         print("Transition for", s_name, a_name, s_next_name, "is", model["T"][a_id].get_prob(s_id, sn))
#         print("Observation for", s_name, a_name, s_next_name, "is", model["Z"][a_id].get_prob(sn, o_id))
