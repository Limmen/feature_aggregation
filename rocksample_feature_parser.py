#!/usr/bin/env python3

import sys
import pickle
import math

###############################################################################
# 1. We'll reuse the SparseRowT / SparseRowO classes exactly as in the original
###############################################################################

class SparseRowT:
    __slots__ = ("num_states", "default_type", "explicit")
    def __init__(self, num_states):
        self.num_states = num_states
        self.default_type = None
        self.explicit = {}

    def set_entire_identity(self):
        self.default_type = "identity"

    def set_entire_uniform(self):
        self.default_type = "uniform"

    def set_sparse_entries(self, s_from, s_to_dict):
        self.explicit[s_from] = s_to_dict

    def set_distribution(self, s_from, dist_list):
        self.explicit[s_from] = dist_list

    def set_row_identity(self, s_from):
        self.explicit[s_from] = ("identity",)

    def set_row_uniform(self, s_from):
        self.explicit[s_from] = ("uniform",)

    def get_prob(self, s_from, s_to):
        row = self.explicit.get(s_from)
        if row is None:
            # fallback
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
            mode = row[0]
            if mode == "identity":
                return 1.0 if (s_to == s_from) else 0.0
            elif mode == "uniform":
                return 1.0 / self.num_states
            else:
                return 0.0
        return 0.0

    def get_successor_states(self, s_from):
        row = self.explicit.get(s_from)
        if row is None:
            # fallback
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
        row = self.explicit.get(s_next)
        if row is None:
            if self.default_type == "identity":
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
        return 0.0

    def get_nonzero_observations(self, s_next):
        row = self.explicit.get(s_next)
        if row is None:
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

###############################################################################
# 2. Helper: Load original model, which presumably uses the same classes
###############################################################################

def load_model(pkl_file):
    """
    Loads the original RockSample .pkl model (with T as a list of SparseRowT,
    Z as a list of SparseRowO, R as a dict, etc.).
    """
    with open(pkl_file, "rb") as f:
        model = pickle.load(f)
    return model

###############################################################################
# 3. Coarsen only the (x,y) portion of each state ID, keep leftover bits & 'st'
###############################################################################

def parse_state_id(state_id, n):
    """
    If state_id == 'st', interpret as terminal => return (None, None, '_TERM').

    Otherwise, we expect something like 'sXY...' with single digits X,Y < n,
    leftover is from index 3 onward.
    """
    if state_id == 'st':
        return (None, None, '_TERM')

    if len(state_id) < 3 or state_id[0] != 's':
        raise ValueError(f"State '{state_id}' not 'st' nor 'sXY...'")

    x_char = state_id[1]
    y_char = state_id[2]
    x = int(x_char)
    y = int(y_char)
    if x >= n or y >= n:
        raise ValueError(f"Parsed x={x}, y={y} >= n={n} from '{state_id}'")

    leftover = state_id[3:]
    return (x, y, leftover)

def coarsen_xy(x, y, x_res, y_res):
    return (x // x_res, y // y_res)

def build_coarsened_state_id(state_id, n, x_res, y_res):
    """
    If 'st', keep 'st'. Otherwise parse (x,y,leftover), coarsen (x,y) => (cx,cy),
    new id = f"sc{cx:02d}{cy:02d}{leftover}".
    """
    if state_id == 'st':
        return 'st'

    x,y,leftover = parse_state_id(state_id, n)
    if x is None and y is None and leftover=='_TERM':
        return 'st'

    cx, cy = coarsen_xy(x,y,x_res,y_res)
    return f"sc{cx:02d}{cy:02d}{leftover}"

###############################################################################
# 4. Build the coarsened model with the same sparse structure
###############################################################################

def build_coarsened_model(orig_model, n, x_res, y_res):
    """
    - All old states -> new coarsened ID.
    - cluster states that share same new ID.
    - build new T (list of SparseRowT), new Z (list of SparseRowO),
      new R (dict) by naive averaging among sub-states in each cluster.
    """

    old_states = orig_model["states"]
    old_actions = orig_model["actions"]
    old_observations = orig_model["observations"]
    old_state_index = orig_model["state_index"]
    old_T_structs = orig_model["T"]  # list of SparseRowT
    old_Z_structs = orig_model["Z"]  # list of SparseRowO
    old_R = orig_model["R"]         # dict {(a_id, s, s_next, obs): reward}
    discount = orig_model["discount"]
    values = orig_model.get("values", None)
    start_dist = orig_model["start"]

    A = len(old_actions)
    S = len(old_states)
    O = len(old_observations)

    # 1) Map old_state -> new_coarse_id, gather clusters
    coarsen_map = {}
    cluster_dict = {}
    for s_id in old_states:
        c_id = build_coarsened_state_id(s_id, n, x_res, y_res)
        coarsen_map[s_id] = c_id
        if c_id not in cluster_dict:
            cluster_dict[c_id] = []
        cluster_dict[c_id].append(s_id)

    # 2) new states = sorted cluster keys
    new_states = sorted(cluster_dict.keys())
    new_state_index = { c_id: i for i,c_id in enumerate(new_states) }
    C = len(new_states)

    # 3) We'll build T_coarse (list of SparseRowT),
    #    Z_coarse (list of SparseRowO),
    #    R_coarse (a dict).
    T_coarse = []
    Z_coarse = []
    for _ in range(A):
        T_coarse.append(SparseRowT(C))
        Z_coarse.append(SparseRowO(C, O))

    R_coarse = {}  # dict {(a_id, cS, cS_next, o): float}

    # We'll do naive uniform weighting among sub-states in each cluster.

    # 3a) Build T_coarse
    # T_coarse[a].set_sparse_entries(cS, dictOf cS'-> prob)
    for a_id in range(A):
        old_T_a = old_T_structs[a_id]  # a SparseRowT for that action
        for cS_id in new_states:
            cS_idx = new_state_index[cS_id]
            sub_states = cluster_dict[cS_id]
            w_sub = 1.0 / len(sub_states)

            # accumulate next_coarse-> prob
            next_map = {}
            for s_old_id in sub_states:
                s_old_idx = old_state_index[s_old_id]
                # get successors
                succ = old_T_a.get_successor_states(s_old_idx)
                for (s_next_idx_old, p_val) in succ:
                    if p_val == 0.0:
                        continue
                    s_next_old_id = old_states[s_next_idx_old]
                    cS_next_id = coarsen_map[s_next_old_id]
                    cS_next_idx = new_state_index[cS_next_id]
                    next_map[cS_next_idx] = next_map.get(cS_next_idx, 0.0) + w_sub*p_val

            if not next_map:
                # no successors => set row to maybe 0 or identity?
                # We'll do nothing => default prob=0
                pass
            else:
                # store as dict
                T_coarse[a_id].set_sparse_entries(cS_idx, next_map)

    # 3b) Build Z_coarse
    # Z_coarse[a].set_sparse_entries(cS_next, dictOf obs-> prob)
    for a_id in range(A):
        old_Z_a = old_Z_structs[a_id]
        for cS_next_id in new_states:
            cS_next_idx = new_state_index[cS_next_id]
            sub_states_next = cluster_dict[cS_next_id]
            w_sub2 = 1.0 / len(sub_states_next)
            obs_map = {}
            # accumulate for each obs
            for o_id in range(O):
                p_sum = 0.0
                for s_next_old_id in sub_states_next:
                    s_next_idx_old = old_state_index[s_next_old_id]
                    p_sum += w_sub2 * old_Z_a.get_prob(s_next_idx_old, o_id)
                if p_sum != 0.0:
                    obs_map[o_id] = p_sum
            if obs_map:
                Z_coarse[a_id].set_sparse_entries(cS_next_idx, obs_map)

    # 3c) Build R_coarse as a dict
    # R_coarse[(a_id, cS, cS_next, o)] = average...
    for a_id in range(A):
        for cS_id in new_states:
            cS_idx = new_state_index[cS_id]
            sub_S = cluster_dict[cS_id]
            w_s = 1.0 / len(sub_S)
            for cS_next_id in new_states:
                cS_next_idx = new_state_index[cS_next_id]
                sub_Sp = cluster_dict[cS_next_id]
                w_sp = 1.0 / len(sub_Sp)
                for o_id in range(O):
                    val_sum = 0.0
                    for s_old_id in sub_S:
                        s_old_idx = old_state_index[s_old_id]
                        for s_next_old_id in sub_Sp:
                            s_next_idx_old = old_state_index[s_next_old_id]
                            r_here = old_R.get((a_id, s_old_idx, s_next_idx_old, o_id), 0.0)
                            val_sum += w_s * w_sp * r_here
                    if val_sum != 0.0:
                        R_coarse[(a_id, cS_idx, cS_next_idx, o_id)] = val_sum

    # 4) Build new start distribution
    old_start = start_dist
    new_start = [0.0]*C
    for i_s, s_old_id in enumerate(old_states):
        p_val = old_start[i_s]
        cS_id = coarsen_map[s_old_id]
        cS_idx = new_state_index[cS_id]
        new_start[cS_idx] += p_val

    # 5) Final dictionary
    new_model = {
        "states": list(new_states),
        "actions": orig_model["actions"],
        "observations": orig_model["observations"],
        "state_index": { s_id: i for i, s_id in enumerate(new_states) },
        "action_index": orig_model["action_index"],
        "obs_index": orig_model["obs_index"],
        "start": new_start,
        "discount": discount,
        "values": values,
        "T": T_coarse,  # list of SparseRowT
        "Z": Z_coarse,  # list of SparseRowO
        "R": R_coarse,  # dict
    }

    return new_model

###############################################################################
# 5. The main script
###############################################################################

def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <orig_model.pkl> <coarse_model.pkl> <n> <x_res> <y_res>")
        print(f"Example: python {sys.argv[0]} rock_4_4.pkl coarse_4_4.pkl 4 2 2")
        sys.exit(1)

    input_model_file = sys.argv[1]
    output_model_file = sys.argv[2]
    n = int(sys.argv[3])       # grid size
    x_res = int(sys.argv[4])   # coarsening factor in x
    y_res = int(sys.argv[5])   # coarsening factor in y

    # 1) load the original model
    orig_model = load_model(input_model_file)

    # 2) build coarsened
    new_model = build_coarsened_model(orig_model, n, x_res, y_res)

    # 3) save
    with open(output_model_file, "wb") as f:
        pickle.dump(new_model, f)

    print(f"Coarsened model saved to {output_model_file}.")

if __name__ == "__main__":
    main()
