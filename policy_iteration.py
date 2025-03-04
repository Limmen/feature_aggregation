import numpy as np
from general_util import GeneralUtil


class PI:
    """
    Implements Howard's Policy Iteration for MDPs
    """

    @staticmethod
    def pi(P, mu, N, gamma, C, X, U, x0, verbose = False):
        """
        Policy iteration
        """
        avg_returns = []
        running_avg_returns = []
        for i in range(0, N):
            avg_return = PI.evaluate_policy(mu, P, C, X, U, x0)
            avg_returns.append(avg_return)
            running_avg_J = GeneralUtil.running_average(avg_returns, 10)
            running_avg_returns.append(running_avg_J)
            if verbose:
                print(f"[PI] i:{i}, avg_return: {avg_return}")

            J = PI.policy_evaluation(P, mu, C, gamma, X, U)
            mu = PI.policy_improvement(P, C, gamma, J, X, U)
        return mu, J

    @staticmethod
    def P_mu(P, mu, X):
        """
        Returns the transition matrix under mu
        """
        P_pi = np.zeros((len(X), len(X)))
        for x in X:
            action = np.where(np.array(mu[x]) == 1)[0][0]
            P_pi[x] = P[action][x]
            assert round(sum(P_pi[x]), 2) == 1
        return P_pi

    @staticmethod
    def C_mu(C, mu, X, U):
        """
        Cost vector under mu
        """
        C_mu = np.zeros((len(X)))
        for x in X:
            C_mu[x] = sum([mu[x][u] * C[x][u] for u in U])
        return C_mu

    @staticmethod
    def policy_evaluation(P, mu, C, gamma, X, U):
        """
        Computes J_mu
        """
        P_mu = PI.P_mu(P, mu, X)
        C_mu = PI.C_mu(C, mu, X, U)
        I = np.identity(len(X))
        J_mu = np.dot(np.linalg.inv(I - (np.dot(gamma, P_mu))), C_mu)
        return np.array(J_mu)

    @staticmethod
    def policy_improvement(P, C, gamma, J, X, U):
        """
        Computes mu'
        """
        mu_prime = np.zeros((len(X), len(U)))
        for x in X:
            control_values = np.zeros(len(U))
            for u in U:
                for x_prime in X:
                    control_values[u] += P[u][x][x_prime] * (C[x][u] + gamma * J[x_prime])
            mu_prime[x][np.argmin(control_values)] = 1
        return mu_prime

    @staticmethod
    def evaluate_policy(mu, P, C, X, U, x0) -> float:
        """
        Estimates J_mu
        """
        returns = []
        for i in range(100):
            x = x0
            Cost = 0
            t = 0
            while t <= 100:
                u = np.random.choice(U, p=mu[x])
                Cost += C[x][u]
                x = int(np.random.choice(X, p=P[u][x]))
                t += 1
            returns.append(Cost)
        avg_return = np.mean(returns)
        return float(avg_return)
