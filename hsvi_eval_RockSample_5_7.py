from pomdp_parser import POMDPParser
from hsvi_eval import parse_pomdp_policy, evaluate_pomdp_policy
import numpy as np

if __name__ == "__main__":
    data = np.load("rocksample_5_7.npz")
    X = data["X"]
    O = data["O"]
    Z = data["Z"]
    C = data["C"]
    P = data["P"]
    U = data["U"]
    b0 = data["b0"]
    policy_dict = parse_pomdp_policy("RockSample_5_7.hsvi")
    # b0 = pomdp.start_dist
    print(b0)
    value, action = evaluate_pomdp_policy(policy_dict, b0)
    print(f"\nBelief: {b0}")
    print(f"Best value:  {value}")
    print(f"Best action: {action}")
