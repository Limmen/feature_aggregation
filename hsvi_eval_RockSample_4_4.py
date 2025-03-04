from pomdp_parser import POMDPParser
from hsvi_eval import parse_pomdp_policy, evaluate_pomdp_policy

if __name__ == "__main__":
    pomdp = POMDPParser.parse_pomdp("RockSample_4_4.pomdp")
    policy_dict = parse_pomdp_policy("RockSample_4_4.hsvi")
    b0 = pomdp.start_dist
    print(b0)
    value, action = evaluate_pomdp_policy(policy_dict, b0)
    print(f"\nBelief: {b0}")
    print(f"Best value:  {value}")
    print(f"Best action: {action}")