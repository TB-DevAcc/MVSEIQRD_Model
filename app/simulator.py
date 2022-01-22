import numpy as np
import scipy
from scipy.integrate import DOP853, RK23, RK45, odeint, solve_ivp


class Simulator:
    def __init__(self) -> None:
        self.ode_list = None
        self.simulation_type = None
        self.params = None
        self.J = 1
        self.K = 1
        self.check_S = []

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
        self.simulation_type = simulation_type
        # TODO make sure attribute error doesn't crash this
        self.J = params["J"]
        self.K = params["K"]
        self.N = params["N"]
        self.N_total = params["N_total"]
        self.B = params["B"]
        self.sigma = params["sigma"]

        self._M0 = params["M"]
        self._V0 = params["V"]
        self._S0 = params["S"]
        self._E_tr0 = params["E_tr"]
        self._E_nt0 = params["E_nt"]
        self._I_asym0 =params["I_asym"]
        self._I_sym0 = params["I_sym"]
        self._I_sev0 = params["I_sev"]
        self._Q_asym0 = params["Q_asym"]
        self._Q_sym0 = params["Q_sym"]
        self._Q_sev0 = params["Q_sev"]
        self._R0 = params["R"]
        self._D0 = params["D"]

        return self._run_ode_system(params)

    def _build_dMdt(self, class_simulation_type: str = "M") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "M":
                rho_mat = self.params["rho_mat"]
                M = self.params["M"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * rho_mat * M)

        return np.array(res).sum(axis=0)

    def _build_dVdt(self, class_simulation_type: str = "V S") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "V":
                rho_vac = self.params["rho_vac"]
                V = self.params["V"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * rho_vac * V)
            if cls == "S":
                nu = self.params["nu"]
                S = self.params["S"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(nu * S)

        return np.array(res).sum(axis=0)

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
                nu = self.params["nu"]
                S = self.params["S"]
                V = self.params["V"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(rho_vac * V - nu * S)
            if cls == "R":
                rho_rec = self.params["rho_rec"]
                R = self.params["R"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(rho_rec * R)
            # Infectious
            if cls == "I3":
                S = self.params["S"]
                N = self.params["N"]
                beta_asym = self.params["beta_asym"]
                beta_sym = self.params["beta_sym"]
                beta_sev = self.params["beta_sev"]
                I_asym = self.params["I_asym"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy

                # TODO choose one alternative and delete the other
                res.append(
                    [
                        np.array(
                            [
                                -np.sum(
                                    [
                                        beta_asym[j, l, k] * I_asym[j, l]
                                        + beta_sym[j, l, k] * I_sym[j, l]
                                        + beta_sev[j, l, k] * I_sev[j, l]
                                        for l in range(self.K)
                                    ]
                                )
                                * S[j, k] / N[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
                # Alternative - [:, l, k] verschiebt Dimensionen
                # res.append(
                #     np.array(
                #         [-np.sum(
                #             [beta_asym[:, l, k] * I_asym[:, l]
                #              + beta_sym[:, l, k] * I_sym[:, l]
                #              + beta_sev[:, l, k] * I_sev[:, l]
                #              for l in range(self.K)]
                #         ) * S[:, k] for k in range(self.K)]
                #     ).swapaxes(0, 1)
                # )

        self.check_S.append(np.array(res).sum(axis=0))
        return np.array(res).sum(axis=0)

    def _build_dE_ntdt(self, class_simulation_type: str = "E2 I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                beta_asym = self.params["beta_asym"]
                S = self.params["S"]
                N = self.params["N"]
                I_asym = self.params["I_asym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    [
                        np.array(
                            [
                                np.sum([beta_asym[j, l, k] * I_asym[j, l] for l in range(self.K)])
                                * S[j, k] / N[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )

                beta_sym = self.params["beta_sym"]
                beta_sev = self.params["beta_sev"]
                psi = self.params["psi"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    [
                        np.array(
                            [
                                np.sum(
                                    [
                                        beta_sym[j, l, k]
                                        * (1 - psi[j, l] * psi[j, k])
                                        * I_sym[j, l]
                                        + beta_sev[j, l, k]
                                        * (1 - psi[j, l] * psi[j, k])
                                        * I_sev[j, l]
                                        for l in range(self.K)
                                    ]
                                )
                                * S[j, k] / N[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
            if cls == "E2":
                epsilon = self.params["epsilon"]
                Ent = self.params["E_nt"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * epsilon * Ent)

        return np.array(res).sum(axis=0)

    def _build_dE_trdt(self, class_simulation_type: str = "E2 I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                beta_sym = self.params["beta_sym"]
                beta_sev = self.params["beta_sev"]
                psi = self.params["psi"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                S = self.params["S"]
                N = self.params["N"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    [
                        np.array(
                            [
                                np.sum(
                                    [
                                        beta_sym[j, l, k] * psi[j, l] * psi[j, k] * I_sym[j, l]
                                        + beta_sev[j, l, k] * psi[j, l] * psi[j, k] * I_sev[j, l]
                                        for l in range(self.K)
                                    ]
                                )
                                * S[j, k] / N[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
            if cls == "E2":
                epsilon = self.params["epsilon"]
                Etr = self.params["E_tr"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * epsilon * Etr)

        return np.array(res).sum(axis=0)

    def _build_dI_asymdt(self, class_simulation_type: str = "E2 I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "E2":
                epsilon = self.params["epsilon"]
                Ent = self.params["E_nt"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(epsilon * Ent)
            if cls == "I3":
                gamma_asym = self.params["gamma_asym"]
                my_sym = self.params["my_sym"]
                my_sev = self.params["my_sev"]
                tau_asym = self.params["tau_asym"]
                I_asym = self.params["I_asym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    -1
                    * (gamma_asym * I_asym + my_sym * I_asym + my_sev * I_asym + tau_asym * I_asym)
                )

        return np.array(res).sum(axis=0)

    def _build_dI_symdt(self, class_simulation_type: str = "I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                my_sym = self.params["my_sym"]
                my_sev = self.params["my_sev"]
                gamma_sym = self.params["gamma_sym"]
                tau_sym = self.params["tau_sym"]
                I_asym = self.params["I_asym"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    my_sym * I_asym
                    + my_sym * I_sev
                    - (gamma_sym * I_sym + my_sev * I_sym + tau_sym * I_sym)
                )

        return np.array(res).sum(axis=0)

    def _build_dI_sevdt(self, class_simulation_type: str = "I3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                my_sev = self.params["my_sev"]
                gamma_sev_r = self.params["gamma_sev_r"]
                gamma_sev_d = self.params["gamma_sev_d"]
                my_sym = self.params["my_sym"]
                tau_sev = self.params["tau_sev"]
                I_asym = self.params["I_asym"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                Q_sev = self.params["Q_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    my_sev * I_asym
                    + my_sev * I_sym
                    - 1
                    * (
                        [
                            np.array(
                                [
                                    (
                                        (1 - self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k]))
                                        * gamma_sev_r[j, k]
                                        + self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k])
                                        * gamma_sev_d[j, k]
                                    )
                                    * I_sev[j, k]
                                    for k in range(self.K)
                                ]
                            )
                            for j in range(self.J)
                        ]
                        + my_sym * I_sev
                        + tau_sev * I_sev
                    )
                )

        return np.array(res).sum(axis=0)

    def _build_dQ_asymdt(self, class_simulation_type: str = "E2 I3 Q3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "E2":
                epsilon = self.params["epsilon"]
                Etr = self.params["E_tr"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(epsilon * Etr)
            if cls == "I3":
                tau_asym = self.params["tau_asym"]
                I_asym = self.params["I_asym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(tau_asym * I_asym)
            if cls == "Q3":
                gamma_asym = self.params["gamma_asym"]
                my_sym = self.params["my_sym"]
                my_sev = self.params["my_sev"]
                Q_asym = self.params["Q_asym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * (gamma_asym * Q_asym + my_sym * Q_asym + my_sev * Q_asym))

        return np.array(res).sum(axis=0)

    def _build_dQ_symdt(self, class_simulation_type: str = "I3 Q3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                tau_sym = self.params["tau_sym"]
                I_sym = self.params["I_sym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(tau_sym * I_sym)
            if cls == "Q3":
                my_sym = self.params["my_sym"]
                gamma_sym = self.params["gamma_sym"]
                my_sev = self.params["my_sev"]
                Q_asym = self.params["Q_asym"]
                Q_sev = self.params["Q_sev"]
                Q_sym = self.params["Q_sym"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(my_sym * Q_asym + my_sym * Q_sev - (gamma_sym * Q_sym + my_sev * Q_sym))

        return np.array(res).sum(axis=0)

    def _build_dQ_sevdt(self, class_simulation_type: str = "I3 Q3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                tau_sev = self.params["tau_sev"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(tau_sev * I_sev)
            if cls == "Q3":
                my_sev = self.params["my_sev"]
                my_sym = self.params["my_sym"]
                gamma_sev_r = self.params["gamma_sev_r"]
                gamma_sev_d = self.params["gamma_sev_d"]
                Q_asym = self.params["Q_asym"]
                Q_sym = self.params["Q_sym"]
                Q_sev = self.params["Q_sev"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    my_sev * Q_asym
                    + my_sev * Q_sym
                    - 1
                    * (
                        [
                            np.array(
                                [
                                    (
                                        (1 - self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k]))
                                        * gamma_sev_r[j, k]
                                        + self._calc_sigma(
                                            j, k, I_sev[j, k], Q_sev[j, k] * gamma_sev_d[j, k]
                                        )
                                    )
                                    * Q_sev[j, k]
                                    for k in range(self.K)
                                ]
                            )
                            for j in range(self.J)
                        ]
                        + my_sym * Q_sev
                    )
                )

        return np.array(res).sum(axis=0)

    def _build_dRdt(self, class_simulation_type: str = "I3 Q3 R") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                gamma_asym = self.params["gamma_asym"]
                gamma_sym = self.params["gamma_sym"]
                gamma_sev_r = self.params["gamma_sev_r"]
                I_asym = self.params["I_asym"]
                I_sym = self.params["I_sym"]
                I_sev = self.params["I_sev"]
                Q_sev = self.params["Q_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    gamma_asym * I_asym
                    + gamma_sym * I_sym
                    + (
                        [
                            np.array(
                                [
                                    1 - self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k])
                                    for k in range(self.K)
                                ]
                            )
                            for j in range(self.J)
                        ]
                    )
                    * gamma_sev_r
                    * I_sev
                )
            if cls == "Q3":
                gamma_asym = self.params["gamma_asym"]
                gamma_sym = self.params["gamma_sym"]
                gamma_sev_r = self.params["gamma_sev_r"]
                Q_asym = self.params["Q_asym"]
                Q_sym = self.params["Q_sym"]
                Q_sev = self.params["Q_sev"]
                I_sev = self.params["I_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    gamma_asym * Q_asym
                    + gamma_sym * Q_sym
                    + (
                        [
                            np.array(
                                [
                                    1 - self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k])
                                    for k in range(self.K)
                                ]
                            )
                            for j in range(self.J)
                        ]
                    )
                    * gamma_sev_r
                    * Q_sev
                )
            if cls == "R":
                rho_rec = self.params["rho_rec"]
                R = self.params["R"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(-1 * rho_rec * R)

        return np.array(res).sum(axis=0)

    def _build_dDdt(self, class_simulation_type: str = "I3 Q3") -> np.ndarray:
        res = []
        for cls in class_simulation_type.split():
            if cls == "I3":
                gamma_sev_d = self.params["gamma_sev_d"]
                I_sev = self.params["I_sev"]
                Q_sev = self.params["Q_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    [
                        np.array(
                            [
                                self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k])
                                * gamma_sev_d[j, k]
                                * I_sev[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
            if cls == "Q3":
                gamma_sev_d = self.params["gamma_sev_d"]
                I_sev = self.params["I_sev"]
                Q_sev = self.params["Q_sev"]
                # TODO check numpy math and make sure it's not a shallow copy
                res.append(
                    [
                        np.array(
                            [
                                self._calc_sigma(j, k, I_sev[j, k], Q_sev[j, k])
                                * gamma_sev_d[j, k]
                                * Q_sev[j, k]
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )

        return np.array(res).sum(axis=0)

    def _build_ode_system(self, t, params: dict) -> list:
        """
        builds an ODE system for a given simulation type.
        E.g. simulation_type = "SI" -> [dSdt = ..., dIdt = ...]

        Returns
        -------
        list
            List of Ordinary differential equations that build a system
        """
        tmp = params.reshape((13, self.J, self.K))
        self.params["M"] = tmp[0]
        self.params["V"] = tmp[1]
        self.params["S"] = tmp[2]
        self.params["E_tr"] = tmp[3]
        self.params["E_nt"] = tmp[4]
        self.params["I_asym"] = tmp[5]
        self.params["I_sym"] = tmp[6]
        self.params["I_sev"] = tmp[7]
        self.params["Q_asym"] = tmp[8]
        self.params["Q_sym"] = tmp[9]
        self.params["Q_sev"] = tmp[10]
        self.params["R"] = tmp[11]
        self.params["D"] = tmp[12]

        print("t: ", t)

        if self.simulation_type == "I3 S E2 I3 Q3 R I" or self.simulation_type == "full":
            return np.array([
                self._build_dMdt(),
                self._build_dVdt(),
                self._build_dSdt(),
                self._build_dE_trdt(),
                self._build_dE_ntdt(),
                self._build_dI_asymdt(),
                self._build_dI_symdt(),
                self._build_dI_sevdt(),
                self._build_dQ_asymdt(),
                self._build_dQ_symdt(),
                self._build_dQ_sevdt(),
                self._build_dRdt(),
                self._build_dDdt(),
            ]).ravel()
        # TODO implement different simulation types

    def _run_ode_system(self, params) -> dict:
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
        return self.simulate_RK45(params["t"], params)

    def _calc_sigma(self, d, k, I_sev_dk, Q_sev_dk):
        """
        Calculates value of sigma for specific situation dependent on available hospital beds
        :param d: index of current district
        :param k: index of current group
        :param I_sev_dk: current value of I_sev for district d in group k
        :param Q_sev_dk: current value of Q_sev for district d in group k
        :return: value of sigma for current situation
        """
        if (I_sev_dk + Q_sev_dk) * self.N[d, k] <= (self.N[d, k] / self.N_total[d]) * self.B[d]:
            return self.sigma[d, k]
        else:
            return (
                self.sigma[d, k] * (self.N[d, k] / self.N_total[d]) * self.B[d]
                + (I_sev_dk + Q_sev_dk) * self.N[d, k]
                - (self.N[d, k] / self.N_total[d]) * self.B[d]
            ) / ((I_sev_dk + Q_sev_dk) * self.N[d, k])

    def simulate_RK45(self, t, params):
        """
        Use solve_ivp with method 'RK45'
        """
        return self._simulate_ivp(t, params, scipy.integrate.RK45)

    def simulate_RK23(self, t, params):
        """
        Use solve_ivp with method 'RK23'
        """
        return self._simulate_ivp(t, params, scipy.integrate.RK23)

    def simulate_DOP853(self, t, params):
        """
        Use solve_ivp with method 'DOP853'
        """
        return self._simulate_ivp(t, params, scipy.integrate.DOP853)

    def _simulate_ivp(self, t, params, method):
        """
        Solve ODE system with solve_ivp
        :param t: timesteps
        :return: solution of ODE system solved with scipy.solve_ivp
        """
        sol = solve_ivp(
            fun=self._build_ode_system,
            t_span=[t[0], t[-1]],
            t_eval=t,
            y0=np.array(
                [
                    self._M0,
                    self._V0,
                    self._S0,
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
            np.array([sol.y[:, i].reshape((13, self.J, self.K))[j] for i in range(len(t))])
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
            np.array([sol[i, :].reshape((13, self.J, self.K))[j] for i in range(len(t))])
            for j in range(13)
        ]

        return result
