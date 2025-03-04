from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser
import numpy as np
from pomdp_parser import POMDPParser
from value_iteration import VI
from pomdp_util import POMDPUtil
import time

if __name__ == '__main__':
    pomdp = POMDPParser.parse_pomdp(file_path="RockSample_5_7.pomdp")
    print(len(pomdp.X))
    print(len(pomdp.A))
    print(len(pomdp.O))
