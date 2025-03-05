from pomdp_parser import POMDPParser
import numpy as np

if __name__ == '__main__':
    pomdp = POMDPParser.parse_pomdp(file_path="RockSample_4_4.pomdp")
    print(pomdp.states)
    # print(pomdp.transition_probs)