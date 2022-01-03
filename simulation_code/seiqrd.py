import numpy as np
from scipy.integrate import solve_ivp, odeint


class SEIQRDModel:

    def __init__(self, D, K, N_total, N, B,
                 S0, E_tr0, E_nt0, I_asym0, I_sym0, I_sev0, Q_asym0, Q_sym0, Q_sev0, R0, D0,
                 alpha, beta_asym, beta_sym, beta_sev, gamma_asym, gamma_sym, gamma_sev,
                 gamma_sev_r, gamma_sev_d, epsilon, eta, nu, psi, sigma):
        """
        Initialize SEIQRD model
        :param D: Number of districts
        :param K: Number of groups
        :param N_total: Number of people in district
        :param N: Number of people in district per group
        :param B: Number of hospital beds in district
        :param alpha: rate of recovered people who are susceptible again
        :param beta_asym:
        :param beta_sym:
        :param beta_sev:
        :param gamma_asym:
        :param gamma_sym:
        :param gamma_sev:
        :param gamma_sev_r:
        :param gamma_sev_d:
        :param epsilon:
        :param eta:
        :param nu:
        :param psi:
        :param sigma:
        """
        self.__D = D
        self.__K = K
        self.__N_total = N_total
        self.__N = N
        self.__B = B
        self.__S0 = S0
        self.__E_tr0 = E_tr0
        self.__E_nt0 = E_nt0
        self.__I_asym0 = I_asym0
        self.__I_sym0 = I_sym0
        self.__I_sev0 = I_sev0
        self.__Q_asym0 = Q_asym0
        self.__Q_sym0 = Q_sym0
        self.__Q_sev0 = Q_sev0
        self.__R0 = R0
        self.__D0 = D0
        self.__alpha = alpha
        self.__beta_asym = beta_asym
        self.__beta_sym = beta_sym
        self.__beta_sev = beta_sev
        self.__gamma_asym = gamma_asym
        self.__gamma_sym = gamma_sym
        self.__gamma_sev = gamma_sev
        self.__gamma_sev_r = gamma_sev_r
        self.__gamma_sev_d = gamma_sev_d
        self.__epsilon = epsilon
        self.__eta = eta
        self.__nu = nu
        self.__psi = psi
        self.__sigma = sigma

    def ode_system(self, t, params):
        """
        Creates ODE system for model
        :param t:
        :param params:
        :return: ODE system
        """
        S, E_nt, E_tr, I_asym, I_sym, I_sev, Q_asym, Q_sym, Q_sev, R, D = params.reshape((11, self.__D, self.__K))

        # S
        dSdt = np.array(
            [np.array(
                [-np.sum([self.__beta_asym[d, l, k] * I_asym[d, l] + self.__beta_sym[d, l, k] * I_sym[d, l]
                          + self.__beta_sev[d, l, k] * I_sev[d, l] for l in range(self.__K)]) * S[d, k] / self.__N[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # E_nt
        dEntdt = np.array(
            [np.array(
                [np.sum([self.__beta_asym[d, l, k] * I_asym[d, l] for l in range(self.__K)]) * S[
                    d, k]  / self.__N[d, k]
                 + np.sum([self.__beta_sym[d, l, k] * (1 - self.__psi[d, l] * self.__psi[d, k]) * I_sym[d, l]
                           + self.__beta_sev[d, l, k] * (1 - self.__psi[d, l] * self.__psi[d, k]) * I_sev[d, l]
                           for l in range(self.__K)]) * S[d, k]  / self.__N[d, k]
                 - self.__epsilon[d, k] * E_nt[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])
        # E_tr
        dEtrdt = np.array(
            [np.array(
                [np.sum([self.__beta_sym[d, l, k] * self.__psi[d, l] * self.__psi[d, k] * I_sym[d, l]
                         + self.__beta_sev[d, l, k] * self.__psi[d, l] * self.__psi[d, k] * I_sev[d, l]
                         for l in range(self.__K)]) * S[d, k]  / self.__N[d, k]
                 - self.__epsilon[d, k] * E_tr[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # I_asym
        dIasymdt = np.array(
            [np.array(
                [self.__eta[d, k] * self.__epsilon[d, k] * E_nt[d, k]
                 - self.__gamma_asym[d, k] * I_asym[d, k] for k in range(self.__K)])
                for d in range(self.__D)])

        # I_sym
        dIsymdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * (1 - self.__nu[d, k]) * self.__epsilon[d, k] * E_nt[d, k]
                 - self.__gamma_sym[d, k] * I_sym[d, k] for k in range(self.__K)])
                for d in range(self.__D)])

        # I_sev
        dIsevdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * self.__nu[d, k] * self.__epsilon[d, k] * E_nt[d, k]
                 - ((1 - self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k])) * self.__gamma_sev_r[d, k]
                    + self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k]) * self.__gamma_sev_d[d, k]) * I_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_asym
        dQasymdt = np.array(
            [np.array(
                [self.__eta[d, k] * self.__epsilon[d, k] * E_tr[d, k]
                 - self.__gamma_asym[d, k] * Q_asym[d, k] for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_sym
        dQsymdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * (1 - self.__nu[d, k]) * self.__epsilon[d, k] * E_tr[d, k]
                 - self.__gamma_sym[d, k] * Q_sym[d, k] for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_sev
        dQsevdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * self.__nu[d, k] * self.__epsilon[d, k] * E_tr[d, k]
                 - ((1 - self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k])) * self.__gamma_sev_r[d, k]
                    + self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k]) * self.__gamma_sev_d[d, k]) * Q_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # R
        dRdt = np.array(
            [np.array(
                [self.__gamma_asym[d, k] * I_asym[d, k] + self.__gamma_sym[d, k] * I_sym[d, k]
                 + (1 - self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k])) * self.__gamma_sev_r[d, k] * I_sev[d, k]
                 + self.__gamma_asym[d, k] * Q_asym[d, k] + self.__gamma_sym[d, k] * Q_sym[d, k]
                 + (1 - self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k])) * self.__gamma_sev_r[d, k] * Q_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # D
        dDdt = np.array(
            [np.array(
                [self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k]) * self.__gamma_sev_d[d, k] * I_sev[d, k]
                 + self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k]) * self.__gamma_sev_d[d, k] * Q_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        return np.array(
            [dSdt, dEntdt, dEtrdt, dIasymdt, dIsymdt, dIsevdt, dQasymdt, dQsymdt, dQsevdt, dRdt, dDdt]).ravel()

    def __calc_sigma(self, d, k, I_sev_dk, Q_sev_dk):
        """
        Calculates value of sigma for specific situation dependent on available hospital beds
        :param d: index of current district
        :param k: index of current group
        :param I_sev_dk: current value of I_sev for district d in group k
        :param Q_sev_dk: current value of Q_sev for district d in group k
        :return: value of sigma for current situation
        """
        if (I_sev_dk + Q_sev_dk) * self.__N[d, k] <= (self.__N[d, k] / self.__N_total[d]) * self.__B[d]:
            return self.__sigma[d, k]
        else:
            return (self.__sigma[d, k] * (self.__N[d, k] / self.__N_total[d]) * self.__B[d] + (
                    I_sev_dk + Q_sev_dk) * self.__N[d, k] - (self.__N[d, k] / self.__N_total[d]) * self.__B[d]) \
                   / ((I_sev_dk + Q_sev_dk) * self.__N[d, k])

    def simulate(self, t):
        """
        Solve ODE system with solve_ivp
        :param t: timesteps
        :return: solution of ODE system solved with scipy.solve_ivp
        """
        sol = solve_ivp(fun=self.ode_system, t_span=[t[0], t[-1]], t_eval=t,
                        y0=np.array(
                            [self.__S0, self.__E_tr0, self.__E_nt0, self.__I_asym0, self.__I_sym0, self.__I_sev0,
                             self.__Q_asym0, self.__Q_sym0, self.__Q_sev0, self.__R0, self.__D0]
                        ).ravel())

        result = [np.array([sol.y[:, i].reshape((11, self.__D, self.__K))[j] for i in range(len(t))]) for j in
                  range(11)]

        return result
