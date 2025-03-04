from pomdp_parser import POMDPParser
import numpy as np

if __name__ == '__main__':
    pomdp = POMDPParser.parse_pomdp(file_path="RockSample_5_5.pomdp")
    X = pomdp.X
    O = pomdp.O
    Z = pomdp.Z
    C = list(-np.array(pomdp.R))
    P = pomdp.T
    U = pomdp.A
    b0 = pomdp.start_dist
    X_arr = np.array(X)
    O_arr = np.array(O)
    Z_arr = np.array(Z)
    C_arr = np.array(C)
    P_arr = np.array(P)
    U_arr = np.array(U)
    np.savez("rocksample_5_5.npz", X=X_arr, O=O_arr, Z=Z_arr, C=C_arr, P=P_arr, U=U_arr, b0=b0)