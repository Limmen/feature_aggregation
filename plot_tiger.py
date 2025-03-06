from csle_tolerance.util.pomdp_solve_parser import PomdpSolveParser
import numpy as np

if __name__ == '__main__':
    alpha_vectors = PomdpSolveParser.parse_alpha_vectors(file_path="tiger.alpha")
    belief_space = np.linspace(0.0, 1, int(1.0 / 0.01))
    for b in belief_space:
        val = np.min([np.dot([1 - b, b], list(-np.array(alpha[1]))) for alpha in alpha_vectors])
        print(f"{b} {val}")
    # val = np.min([np.dot(b0, list(-np.array(alpha[1]))) for alpha in alpha_vectors])