import numpy as np
from scipy.integrate import BDF, DOP853, LSODA, RK23, RK45, Radau, odeint, solve_ivp


class Simulator:
    def __init__(self) -> None:
        self.ode_list = None
        self.simulation_type = None
        self.params = None
        self.J = 0
        self.K = 0

    def run(self, params, simulation_type) -> dict:
        """
        Runs the simulation with the given simulation type

        Parameters
        ----------
        params : dict
            Parameters for the simulation
        simulation_type : str
            Type of simulation to be run e.g. "S I", "S E I R", "I3 S E2 I3 Q3 R I"

        Returns
        -------
        dict
            New parameters
        """
        self.params = params
        # TODO make sure attribute error doesn't crash this
        self.J = params["J"]
        self.K = params["K"]
        self.ode_list = self._build_ode_system(simulation_type)
        return self._run_ode_system(self.ode_list, params)

    def _build_dSdt(self, class_simulation_type: str = "M V R I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            # Immune
            if cls == "M":
                rho_mat = self.params["rho_mat"]
                M = self.params["M"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(rho_mat * M)
            if cls == "V":
                rho_vac = self.params["rho_vac"]
                V = self.params["V"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(rho_vac * V)
            if cls == "R":
                rho_rec = self.params["rho_rec"]
                R = self.params["R"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(rho_rec * R)
            # Infectious
            if cls == "I3":
                S = self.params["S"]
                beta_asym = self.params["beta_asym"]
                beta_sym = self.params["beta_sym"]
                beta_sev = self.params["beta_sev"]
                I_asym = self.params["I_asym"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                I_asym = beta_asym * I_asym
                I_sym = beta_sym * I_sym
                I_sev = beta_sev * I_sev
                res.append(-1 * (I_asym + I_sym + I_sev) * S)
            # TODO other classes: I

        return res.sum()  # TODO sum up all classes

    def _build_dMdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dVdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dE_ntdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dE_trdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dI_asymdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dI_symdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dI_sevdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dQ_asymdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dQ_symdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dQ_sevdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dRdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_dDdt(self, class_simulation_type="") -> np.ndarray:
        # TODO implement
        ...

    def _build_ode_system(self, simulation_type: str) -> list:
        """
        builds an ODE system for a given simulation type.
        E.g. simulation_type = "SI" -> [dSdt = ..., dIdt = ...]

        Returns
        -------
        list
            List of Ordinary differential equations that build a system
        """

        if simulation_type == "I3 S E2 I3 Q3 R I" or simulation_type == "full":
            return [
                self._build_dMdt(),
                self._build_dVdt(),
                self._build_dSdT(),
                self._build_dE_ntdt(),
                self._build_dE_trdt(),
                self._build_dI_asymdt(),
                self._build_dI_symdt(),
                self._build_dI_sevdt(),
                self._build_dQ_asymdt(),
                self._build_dQ_symdt(),
                self._build_dQ_sevdt(),
                self._build_dRdt(),
                self._build_dDdt(),
            ]

        # TODO implement different simulation types

    def _run_ode_system(self, ode_list, params) -> dict:
        """
        Creates ODE system for epidemiological model

        Parameters
        ----------
        ode_list : list
            List of single ODEs to be integrated into the system
        params : dict
            Parameters for the simulation

        Returns
        -------
        dict
            New parameters
        """
        # TODO run system
        ...

    def _calc_sigma(self, d, k, I_sev_dk, Q_sev_dk):
        """
        Calculates value of sigma for specific situation dependent on available hospital beds
        :param d: index of current district
        :param k: index of current group
        :param I_sev_dk: current value of I_sev for district d in group k
        :param Q_sev_dk: current value of Q_sev for district d in group k
        :return: value of sigma for current situation
        """
        if (I_sev_dk + Q_sev_dk) * self._N[d, k] <= (self._N[d, k] / self._N_total[d]) * self._B[
            d
        ]:
            return self._sigma[d, k]
        else:
            return (
                self._sigma[d, k] * (self._N[d, k] / self._N_total[d]) * self._B[d]
                + (I_sev_dk + Q_sev_dk) * self._N[d, k]
                - (self._N[d, k] / self._N_total[d]) * self._B[d]
            ) / ((I_sev_dk + Q_sev_dk) * self._N[d, k])

    def simulate_RK45(self, t):
        """
        Use solve_ivp with method 'RK45'
        """
        return self._simulate_ivp(t, RK45)

    def simulate_RK23(self, t):
        """
        Use solve_ivp with method 'RK23'
        """
        return self._simulate_ivp(t, RK23)

    def simulate_DOP853(self, t):
        """
        Use solve_ivp with method 'DOP853'
        """
        return self._simulate_ivp(t, DOP853)

    def simulate_BDF(self, t):
        """
        Use solve_ivp with method 'BDF'
        """
        return self._simulate_ivp(t, BDF)

    def simulate_Radau(self, t):
        """
        Use solve_ivp with method 'Radau'
        """
        return self._simulate_ivp(t, Radau)

    def simulate_LSODA(self, t):
        """
        Use solve_ivp with method 'LSODA'
        """
        return self._simulate_ivp(t, LSODA)

    def _simulate_ivp(self, ode_system, t, method):
        """
        Solve ODE system with solve_ivp
        :param t: timesteps
        :return: solution of ODE system solved with scipy.solve_ivp
        """
        sol = solve_ivp(
            fun=ode_system,
            t_span=[t[0], t[-1]],
            t_eval=t,
            y0=np.array(
                [
                    self._M0,
                    self._S0,
                    self._V0,
                    self._E_tr0,
                    self._E_nt0,
                    self._I_asym0,
                    self._I_sym0,
                    self._I_sev0,
                    self._Q_asym0,
                    self._Q_sym0,
                    self._Q_sev0,
                    self._R0,
                    self._D0,
                ]
            ).ravel(),
            method=method,
        )

        result = [
            np.array([sol.y[:, i].reshape((13, self._D, self._K))[j] for i in range(len(t))])
            for j in range(13)
        ]

        return result

    def simulate_odeint(self, ode_system, t):
        """
        Solve ODE system with odeint
        :param t: timesteps
        :return: solution of ODE system solved with scipy.odeint
        """
        sol = odeint(
            func=ode_system,
            t=t,
            y0=np.array(
                [
                    self._M0,
                    self._S0,
                    self._V0,
                    self._E_tr0,
                    self._E_nt0,
                    self._I_asym0,
                    self._I_sym0,
                    self._I_sev0,
                    self._Q_asym0,
                    self._Q_sym0,
                    self._Q_sev0,
                    self._R0,
                    self._D0,
                ]
            ).ravel(),
            tfirst=True,
        )

        result = [
            np.array([sol[i, :].reshape((13, self._D, self._K))[j] for i in range(len(t))])
            for j in range(13)
        ]

        return result
