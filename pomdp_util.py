import time
import random
import math
from multiprocessing import Pool
import numpy as np
import itertools
from collections import Counter


class POMDPUtil:
    """
    Utility functions for belief aggregation in POMDPs
    """

    @staticmethod
    def belief_operator(z, u, b, X, Z, P):
        """
        Computes b' after observing (b,o)
        """
        b_prime = [0.0] * len(X)
        for x_prime in X:
            b_prime[x_prime] = POMDPUtil.bayes_filter(
                x_prime=x_prime, z=z, u=u, b=b, X=X, P=P, Z=Z)
        assert round(sum(b_prime), 2) == 1
        return b_prime

    @staticmethod
    def bayes_filter(x_prime, z, u, b, X, Z, P):
        """
        A Bayesian filter to compute b[x_prime] after observing (z,u)
        """
        norm = 0.0
        for x in X:
            for x_prime_1 in X:
                prob_1 = Z[u][x_prime_1][z]
                norm += b[x] * prob_1 * P[u][x][x_prime_1]
        temp = 0.0
        for x in X:
            temp += Z[u][x_prime][z] * P[u][x][x_prime] * b[x]
        b_prime_s_prime = temp / norm
        assert round(b_prime_s_prime, 2) <= 1
        return float(b_prime_s_prime)

    @staticmethod
    def nearest_neighbor(B_n, b):
        """
        Returns the nearest neighbor of b in B_n
        """
        distances = np.linalg.norm(np.array(B_n) - np.array(b), axis=1)
        nearest_index = int(np.argmin(distances))
        return B_n[nearest_index]

    @staticmethod
    def B_n(n, X):
        """
        Creates the aggregate belief space B_n, where n is the resolution
        """
        combinations = [k for k in itertools.product(range(n + 1), repeat=len(X)) if sum(k) == n]
        belief_points = [list(float(k_i / n) for k_i in k) for k in combinations]
        return belief_points

    @staticmethod
    def b_0_n(b0, B_n):
        """
        Creates the initial  belief
        """
        return POMDPUtil.nearest_neighbor(B_n=B_n, b=b0)

    @staticmethod
    def C_b(B_n, X, U, C, O, P):
        """
        Generates a cost tensor for the aggregate belief MDP
        """
        belief_C = list(np.zeros((len(B_n), len(U))).tolist())
        for u in U:
            for z in O:
                for b in B_n:
                    belief_C[B_n.index(b)][u] = POMDPUtil.expected_cost(b=b, u=u, C=C, X=X, z=z, P=P)
        return belief_C

    @staticmethod
    def expected_cost(b, u, z, C, X, P):
        """
        Computes E[C[x][u] | b]
        """
        # R[a][s][s'][o]
        return sum([C[u][x][x_prime][z] * b[x] * P[u][x][x_prime] for x in X for x_prime in X])

    @staticmethod
    def P_z_b_u(b, z, Z, X, U, P, u):
        """
        Computes P(z | b, u)
        """
        return sum([Z[x_prime][z] * b[x] * P[u][x][x_prime] for x in X for x_prime in X])

    @staticmethod
    def P_b(B_n, X, U, O, P, Z):
        """
        Generates an aggregate belief space transition operator
        """
        belief_T = list(np.zeros((len(U), len(B_n), len(B_n))).tolist())
        for u in U:
            for b1 in B_n:
                for b2 in B_n:
                    belief_T[u][B_n.index(b1)][B_n.index(b2)] = \
                        POMDPUtil.P_b2_b1_u(b1=b1, b2=b2, u=u, X=X, O=O, P=P, Z=Z, B_n=B_n)
        return belief_T

    @staticmethod
    def P_b2_b1_u(b1, b2, u, X, O, P, Z, B_n):
        """
        Calculates P(b2 | b1, u)
        """
        prob = 0
        for z in O:
            if sum([Z[u][s_prime][z] * b1[s] * P[u][s][s_prime] for s in X for s_prime in X]) == 0:
                continue
            b_prime = POMDPUtil.belief_operator(z=z, u=u, b=b1, X=X, Z=Z, P=P)
            nearest_neighbor = POMDPUtil.nearest_neighbor(B_n=B_n, b=b_prime)
            if nearest_neighbor == b2:
                for x in X:
                    for x_prime in X:
                        prob += Z[u][x_prime][z] * b1[x] * P[u][x][x_prime]
        return prob

    @staticmethod
    def particle_filter(particles, max_num_particles, P, o, u, X, O, Z):
        """
        Implements a particle filter
        """
        new_particles = []
        while len(new_particles) < max_num_particles:
            x = random.choice(particles)
            x_prime = np.random.choice(X, p=P[u][x])
            o_hat = np.random.choice(O, p=Z[u][x_prime])
            if o == o_hat:
                new_particles.append(x_prime)
        return new_particles

    @staticmethod
    def policy_evaluation(L, mu, gamma, b0, P, C, Z, X, N, B_n, O):
        """
        Monte-Carlo evaluation of a policy
        """
        costs = []
        for l in range(L):
            x = np.random.choice(X, p=b0)
            b = b0
            Cost = 0
            for k in range(N):
                b_idx = B_n.index(POMDPUtil.nearest_neighbor(B_n=B_n, b=b))
                u = np.argmax(mu[b_idx])
                x_prime = np.random.choice(X, p=P[u][x])
                z = np.random.choice(O, p=Z[u][x_prime])
                Cost += math.pow(gamma, k) * C[u][x][x_prime][z]
                x = x_prime
                b = POMDPUtil.belief_operator(z=z, u=u, b=b, X=X, Z=Z, P=P)
            costs.append(Cost)
        return np.mean(costs)
