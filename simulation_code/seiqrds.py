import numpy as np
from scipy.integrate import solve_ivp, odeint, RK45, RK23, DOP853, BDF, Radau, LSODA


class SEIQRDSModel:

    def __init__(self, D, K, N_total, N, B,
                 M0, S0, V0, E_tr0, E_nt0, I_asym0, I_sym0, I_sev0, Q_asym0, Q_sym0, Q_sev0, R0, D0,
                 beta_asym, beta_sym, beta_sev, gamma_asym, gamma_sym, gamma_sev, gamma_sev_r, gamma_sev_d,
                 gamma_asym_sym, gamma_asym_sev, gamma_asym_q, gamma_sym_sev, gamma_sev_sym, gamma_sev_q,
                 epsilon, my, ny, rho,
                 eta, xi, psi, sigma, tau, kappa):
        """
        Initialize SEIQRDS model
        :param D: Number of districts
        :param K: Number of groups
        :param N_total: Number of people in district
        :param N: Number of people in district per group
        :param B: Number of hospital beds in district
        :param M0: Number of born individuals
        :param S0:
        :param V0: Number of vaccinated individuals
        :param E_tr0:
        :param E_nt0:
        :param I_asym0:
        :param I_sym0:
        :param I_sev0:
        :param Q_asym0:
        :param Q_sym0:
        :param Q_sev0:
        :param R0:
        :param D0:
        :param beta_asym:
        :param beta_sym:
        :param beta_sev:
        :param gamma_asym:
        :param gamma_sym:
        :param gamma_sev:
        :param gamma_sev_r:
        :param gamma_sev_d:
        :param gamma_asym_sym: {I/Q}_asym -> {I/Q}_sym
        :param gamma_asym_sev: {I/Q}_asym -> {I/Q}_sev
        :param gamma_asym_q: I_asym -> Q_asym
        :param gamma_sym_sev: {I/Q}_sym -> {I/Q}_sev
        :param gamma_sev_sym: {I/Q}_sev -> {I/Q}_sym
        :param gamma_sev_q: I_sev -> Q_sev
        :param epsilon: latency time
        :param my: immunity after birth
        :param ny: immunity after vaccination
        :param rho: immunity after recovering
        :param eta: rate of asymptomatic infectious individuals
        :param xi: rate of severely symptomatic infectious individuals
        :param psi: rate of individuals using tracing app
        :param sigma: death rate of severely infections
        :param tau:  rate of recovered individuals who are susceptible again
        :param kappa: rate of vaccination
        """
        self.__D = D
        self.__K = K
        self.__N_total = N_total
        self.__N = N
        self.__B = B
        self.__M0 = M0
        self.__S0 = S0
        self.__V0 = V0
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
        self.__beta_asym = beta_asym
        self.__beta_sym = beta_sym
        self.__beta_sev = beta_sev
        self.__gamma_asym = gamma_asym
        self.__gamma_sym = gamma_sym
        self.__gamma_sev = gamma_sev
        self.__gamma_sev_r = gamma_sev_r
        self.__gamma_sev_d = gamma_sev_d
        self.__gamma_asym_sym = gamma_asym_sym
        self.__gamma_asym_sev = gamma_asym_sev
        self.__gamma_asym_q = gamma_asym_q
        self.__gamma_sym_sev = gamma_sym_sev
        self.__gamma_sev_sym = gamma_sev_sym
        self.__gamma_sev_q = gamma_sev_q
        self.__epsilon = epsilon
        self.__eta = eta
        self.__my = my
        self.__ny = ny
        self.__rho = rho
        self.__xi = xi
        self.__psi = psi
        self.__sigma = sigma
        self.__tau = tau
        self.__kappa = kappa

    def ode_system(self, t, params):
        """
        Creates ODE system for model
        :param t:
        :param params:
        :return: ODE system
        """
        M, S, V, E_nt, E_tr, I_asym, I_sym, I_sev, Q_asym, Q_sym, Q_sev, R, D = params.reshape((13, self.__D, self.__K))

        # M
        dMdt = np.array(
            [np.array(
                [- self.__my[d, k] * M[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # S
        dSdt = np.array(
            [np.array(
                [self.__my[d, k] * M[d, k] + self.__rho[d, k] * R[d, k]
                 + self.__ny[d, k] * V[d, k] - self.__kappa[d, k] * S[d, k]
                 - np.sum([self.__beta_asym[d, l, k] * I_asym[d, l] + self.__beta_sym[d, l, k] * I_sym[d, l]
                           + self.__beta_sev[d, l, k] * I_sev[d, l] for l in range(self.__K)]) * S[d, k] / self.__N[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # V
        dVdt = np.array(
            [np.array(
                [self.__kappa[d, k] * S[d, k] - self.__ny[d, k] * V[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # E_nt
        dEntdt = np.array(
            [np.array(
                [np.sum([self.__beta_asym[d, l, k] * I_asym[d, l] for l in range(self.__K)]) * S[d, k] / self.__N[d, k]
                 + np.sum([self.__beta_sym[d, l, k] * (1 - self.__psi[d, l] * self.__psi[d, k]) * I_sym[d, l]
                           + self.__beta_sev[d, l, k] * (1 - self.__psi[d, l] * self.__psi[d, k]) * I_sev[d, l]
                           for l in range(self.__K)]) * S[d, k] / self.__N[d, k]
                 - self.__epsilon[d, k] * E_nt[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])
        # E_tr
        dEtrdt = np.array(
            [np.array(
                [np.sum([self.__beta_sym[d, l, k] * self.__psi[d, l] * self.__psi[d, k] * I_sym[d, l]
                         + self.__beta_sev[d, l, k] * self.__psi[d, l] * self.__psi[d, k] * I_sev[d, l]
                         for l in range(self.__K)]) * S[d, k] / self.__N[d, k]
                 - self.__epsilon[d, k] * E_tr[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # I_asym
        dIasymdt = np.array(
            [np.array(
                [self.__eta[d, k] * self.__epsilon[d, k] * E_nt[d, k]
                 - self.__gamma_asym[d, k] * I_asym[d, k]
                 - self.__gamma_asym_sym[d, k] * I_asym[d, k]
                 - self.__gamma_asym_sev[d, k] * I_asym[d, k]
                 - self.__gamma_asym_q[d, k] * I_asym[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # I_sym
        dIsymdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * (1 - self.__xi[d, k]) * self.__epsilon[d, k] * E_nt[d, k]
                 - self.__gamma_sym[d, k] * I_sym[d, k]
                 + self.__gamma_asym_sym[d, k] * I_asym[d, k]
                 - self.__gamma_sym_sev[d, k] * I_sym[d, k]
                 + self.__gamma_sev_sym[d, k] * I_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # I_sev
        dIsevdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * self.__xi[d, k] * self.__epsilon[d, k] * E_nt[d, k]
                 + self.__gamma_asym_sev[d, k] * I_asym[d, k]
                 + self.__gamma_sym_sev[d, k] * I_sym[d, k]
                 - self.__gamma_sev_sym[d, k] * I_sev[d, k]
                 - self.__gamma_sev_q[d, k] * I_sev[d, k]
                 - ((1 - self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k])) * self.__gamma_sev_r[d, k]
                    + self.__calc_sigma(d, k, I_sev[d, k], Q_sev[d, k]) * self.__gamma_sev_d[d, k]) * I_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_asym
        dQasymdt = np.array(
            [np.array(
                [self.__eta[d, k] * self.__epsilon[d, k] * E_tr[d, k]
                 - self.__gamma_asym[d, k] * Q_asym[d, k]
                 - self.__gamma_asym_sym[d, k] * Q_asym[d, k]
                 - self.__gamma_asym_sev[d, k] * Q_asym[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_sym
        dQsymdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * (1 - self.__xi[d, k]) * self.__epsilon[d, k] * E_tr[d, k]
                 - self.__gamma_sym[d, k] * Q_sym[d, k]
                 - self.__gamma_sym_sev[d, k] * Q_sym[d, k]
                 + self.__gamma_asym_sym[d, k] * Q_asym[d, k]
                 + self.__gamma_sev_sym[d, k] * Q_sev[d, k]
                 for k in range(self.__K)])
                for d in range(self.__D)])

        # Q_sev
        dQsevdt = np.array(
            [np.array(
                [(1 - self.__eta[d, k]) * self.__xi[d, k] * self.__epsilon[d, k] * E_tr[d, k]
                 + self.__gamma_asym_sev[d, k] * Q_asym[d, k]
                 + self.__gamma_sym_sev[d, k] * Q_sym[d, k]
                 + self.__gamma_sev_q[d, k] * I_sev[d, k]
                 - self.__gamma_sev_sym[d, k] * Q_sev[d, k]
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
                 - self.__rho[d, k] * R[d, k]
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
            [dMdt, dSdt, dVdt, dEntdt, dEtrdt, dIasymdt, dIsymdt, dIsevdt, dQasymdt, dQsymdt, dQsevdt, dRdt, dDdt]
        ).ravel()

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

    def simulate_RK45(self, t):
        """
        Use solve_ivp with method 'RK45'
        """
        return self.__simulate_ivp(t, RK45)

    def simulate_RK23(self, t):
        """
        Use solve_ivp with method 'RK23'
        """
        return self.__simulate_ivp(t, RK23)

    def simulate_DOP853(self, t):
        """
        Use solve_ivp with method 'DOP853'
        """
        return self.__simulate_ivp(t, DOP853)

    def simulate_BDF(self, t):
        """
        Use solve_ivp with method 'BDF'
        """
        return self.__simulate_ivp(t, BDF)

    def simulate_Radau(self, t):
        """
        Use solve_ivp with method 'Radau'
        """
        return self.__simulate_ivp(t, Radau)

    def simulate_LSODA(self, t):
        """
        Use solve_ivp with method 'LSODA'
        """
        return self.__simulate_ivp(t, LSODA)

    def __simulate_ivp(self, t, method):
        """
        Solve ODE system with solve_ivp
        :param t: timesteps
        :return: solution of ODE system solved with scipy.solve_ivp
        """
        sol = solve_ivp(fun=self.ode_system, t_span=[t[0], t[-1]], t_eval=t,
                        y0=np.array(
                            [self.__M0, self.__S0, self.__V0, self.__E_tr0, self.__E_nt0,
                             self.__I_asym0, self.__I_sym0, self.__I_sev0,
                             self.__Q_asym0, self.__Q_sym0, self.__Q_sev0, self.__R0, self.__D0]
                        ).ravel(), method=method)

        result = [np.array([sol.y[:, i].reshape((13, self.__D, self.__K))[j] for i in range(len(t))]) for j in
                  range(13)]

        return result

    def simulate_odeint(self, t):
        """
        Solve ODE system with odeint
        :param t: timesteps
        :return: solution of ODE system solved with scipy.odeint
        """
        sol = odeint(func=self.ode_system, t=t, y0=np.array(
            [self.__M0, self.__S0, self.__V0, self.__E_tr0, self.__E_nt0,
             self.__I_asym0, self.__I_sym0, self.__I_sev0,
             self.__Q_asym0, self.__Q_sym0, self.__Q_sev0, self.__R0, self.__D0]
        ).ravel(), tfirst=True)

        result = [np.array([sol[i, :].reshape((13, self.__D, self.__K))[j] for i in range(len(t))]) for j in
                  range(13)]

        return result
