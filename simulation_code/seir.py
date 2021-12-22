import numpy as np
from scipy.integrate import solve_ivp


class SEIRModel:

    def __init__(self, D, N, S0, E0, I0, R0, alpha, beta, gamma):
        """
        Initialize SEIR-Model
        :param D: Number of districts
        :param N:
        :param S0:
        :param E0:
        :param I0:
        :param R0:
        :param alpha:
        :param beta:
        :param gamma:
        """
        self.__D = D
        self.__N = N
        self.__S0 = S0
        self.__E0 = E0
        self.__I0 = I0
        self.__R0 = R0
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma

    def ode_system(self, t, params):
        """
        Creates ODE system for model
        :param t:
        :param params:
        :return: ODE system
        """
        S, E, I, R = params.reshape((4, self.__D))

        dSdt = -self.__beta * S * I / self.__N
        dEdt = self.__beta * S * I / self.__N - self.__alpha * E
        dIdt = self.__alpha * E - self.__gamma * I
        dRdt = self.__gamma * I

        return np.array([dSdt, dEdt, dIdt, dRdt]).ravel()

    def simulate(self, t):
        """
        Solve ode system with solve_ivp
        :param t:
        :return: solution of ode system solved with scipy.solve_ivp
        """
        sol = solve_ivp(fun=self.ode_system, t_span=[t[0], t[-1]], t_eval=t,
                        y0=np.array([self.__S0, self.__E0, self.__I0, self.__R0]).ravel())

        result = [np.array([sol.y[:, i].reshape((4, self.__D))[j] for i in range(len(t))]) for j in range(4)]

        return result
