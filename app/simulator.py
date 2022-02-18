from typing import Tuple

import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp


class Simulator:
    def __init__(self) -> None:
        self.ode_list = None
        self.supported_sim_types = [
            "S I",
            "M V S E2 I3 Q3 R D",
        ]
        self.simulation_type = None
        self.params = None
        self.J = 1
        self.K = 1

    def run(self, params, simulation_type) -> dict:
        """
        Runs the simulation with the given simulation type

        Parameters
        ----------
        params : dict
            Parameters for the simulation
        simulation_type : str
            Type of simulation to be run e.g. "S I", "S E I R", "M V S E2 I3 Q3 R D"

        Returns
        -------
        dict
            New parameters
        """
        self.params = params
        if simulation_type == "full":
            self.simulation_type = "M V S E2 I3 Q3 R D"
        else:
            self.simulation_type = simulation_type
        # TODO make sure attribute error doesn't crash this
        self.J = params["J"]
        self.K = params["K"]
        self.param_count = self._count_params()

        return self._run_ode_system(params)

    def _count_params(self):
        if self.simulation_type == "S I":
            return 2
        if self.simulation_type == "S E I R":
            return 4
        if self.simulation_type == "M V S E2 I3 Q3 R D":
            return 13

    def _build_dMdt(self, t) -> np.ndarray:
        """
        Set up equation for M (newborn)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of M for current iteration
        """
        res = []
        rho_mat = self.params["rho_mat"]
        M = self.params["M"]
        res.append(-1 * rho_mat[int(t)] * M)

        return np.array(res).sum(axis=0)

    def _build_dVdt(self, t) -> np.ndarray:
        """
        Set up equation for V (vaccinated)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of V for current iteration
        """
        res = []

        rho_vac = self.params["rho_vac"]
        V = self.params["V"]
        res.append(-1 * rho_vac[int(t)] * V)

        nu = self.params["nu"]
        S = self.params["S"]
        res.append(nu[int(t)] * S)

        return np.array(res).sum(axis=0)

    def _build_dSdt(self, t) -> np.ndarray:
        """
        Set up equation for S (susceptable)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of S for current iteration
        """
        res = []

        cls = self.simulation_type.split()
        if "M" in cls:
            rho_mat = self.params["rho_mat"]
            M = self.params["M"]
            res.append(rho_mat[int(t)] * M)
        if "V" in cls:
            rho_vac = self.params["rho_vac"]
            nu = self.params["nu"]
            S = self.params["S"]
            V = self.params["V"]
            res.append(rho_vac[int(t)] * V - nu[int(t)] * S)
        if "R" in cls:
            rho_rec = self.params["rho_rec"]
            R = self.params["R"]
            res.append(rho_rec[int(t)] * R)
        if "I3" in cls:
            S = self.params["S"]
            N = self.params["N"]
            beta_asym = self.params["beta_asym"]
            beta_sym = self.params["beta_sym"]
            beta_sev = self.params["beta_sev"]
            I_asym = self.params["I_asym"]
            I_sym = self.params["I_sym"]
            I_sev = self.params["I_sev"]

            res.append(
                [
                    np.array(
                        [
                            -np.sum(
                                [
                                    beta_asym[int(t), j, l, k] * I_asym[j, l]
                                    + beta_sym[int(t), j, l, k] * I_sym[j, l]
                                    + beta_sev[int(t), j, l, k] * I_sev[j, l]
                                    for l in range(self.K)
                                ]
                            )
                            * S[j, k]
                            / N[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
            )
        elif "I2" in cls:
            pass
        elif "I" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_dEdt(self, t) -> Tuple:
        """
        Wrap equations for E (exposed)
        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        Tuple
            Equation for class E based on simulation_type

        """
        cls = self.simulation_type.split()
        if "E2" in cls:
            return self._build_dE_trdt(t), self._build_dE_ntdt(t)
        elif "E" in cls:
            pass

    def _build_dE_ntdt(self, t) -> np.ndarray:
        """
        Set up equation for Ent (exposed and not tracked)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Ent for current iteration
        """
        res = []

        cls = self.simulation_type.split()
        if "I3" in cls:
            beta_asym = self.params["beta_asym"]
            S = self.params["S"]
            N = self.params["N"]
            I_asym = self.params["I_asym"]
            res.append(
                [
                    np.array(
                        [
                            np.sum(
                                [
                                    beta_asym[int(t), j, l, k] * I_asym[j, l]
                                    for l in range(self.K)
                                ]
                            )
                            * S[j, k]
                            / N[j, k]
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
            res.append(
                [
                    np.array(
                        [
                            np.sum(
                                [
                                    beta_sym[int(t), j, l, k]
                                    * (1 - psi[int(t), j, l] * psi[int(t), j, k])
                                    * I_sym[j, l]
                                    + beta_sev[int(t), j, l, k]
                                    * (1 - psi[int(t), j, l] * psi[int(t), j, k])
                                    * I_sev[j, l]
                                    for l in range(self.K)
                                ]
                            )
                            * S[j, k]
                            / N[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
            )
        elif "I" in cls:
            pass

        # method _build_dE_ntdt is only called if E2 is in simulation_type so this has to be executed anways
        epsilon = self.params["epsilon"]
        Ent = self.params["E_nt"]
        res.append(-1 * epsilon[int(t)] * Ent)

        return np.array(res).sum(axis=0)

    def _build_dE_trdt(self, t) -> np.ndarray:
        """
        Set up equation for Etr (exposed and tracked)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Etr for current iteration
        """
        res = []

        cls = self.simulation_type.split()
        if "I3" in cls:
            beta_sym = self.params["beta_sym"]
            beta_sev = self.params["beta_sev"]
            psi = self.params["psi"]
            I_sym = self.params["I_sym"]
            I_sev = self.params["I_sev"]
            S = self.params["S"]
            N = self.params["N"]
            res.append(
                [
                    np.array(
                        [
                            np.sum(
                                [
                                    beta_sym[int(t), j, l, k]
                                    * psi[int(t), j, l]
                                    * psi[int(t), j, k]
                                    * I_sym[j, l]
                                    + beta_sev[int(t), j, l, k]
                                    * psi[int(t), j, l]
                                    * psi[int(t), j, k]
                                    * I_sev[j, l]
                                    for l in range(self.K)
                                ]
                            )
                            * S[j, k]
                            / N[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
            )
        elif "I2" in cls:
            pass

        # method _build_dE_trdt is only called if E2 is in simulation_type so this has to be executed anways
        epsilon = self.params["epsilon"]
        Etr = self.params["E_tr"]
        res.append(-1 * epsilon[int(t)] * Etr)

        return np.array(res).sum(axis=0)

    def _build_dIdt(self, t) -> Tuple:
        """
        Wrap equation for I (infectious)
        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        Tuple
            Equation for class I based on simulation_type

        """
        cls = self.simulation_type.split()
        if "I3" in cls and "Q3" in cls:
            return (
                self._build_dI_asymdt(t),
                self._build_dI_symdt(t),
                self._build_dI_sevdt(t),
            )
        elif "I2" in cls:
            return self._build_dI_asymdt(t), self._build_dI_symdt(t)
        elif "I" in cls:
            pass

    def _build_dI_asymdt(self, t) -> np.ndarray:
        """
        Set up equation for Iasym (asymptomatic infectious)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Iasym for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "E2" in cls:
            epsilon = self.params["epsilon"]
            Ent = self.params["E_nt"]
            res.append(epsilon[int(t)] * Ent)
        if "I3" in cls:
            gamma_asym = self.params["gamma_asym"]
            mu_sym = self.params["mu_sym"]
            mu_sev = self.params["mu_sev"]
            tau_asym = self.params["tau_asym"]
            I_asym = self.params["I_asym"]
            res.append(
                -1
                * (
                    gamma_asym[int(t)] * I_asym
                    + mu_sym[int(t)] * I_asym
                    + mu_sev[int(t)] * I_asym
                    + tau_asym[int(t)] * I_asym
                )
            )
        elif "I2" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_dI_symdt(self, t) -> np.ndarray:
        """
        Set up equation for Isym (symptomatic infectious)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Isym for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "I3" in cls:
            mu_sym = self.params["mu_sym"]
            mu_sev = self.params["mu_sev"]
            gamma_sym = self.params["gamma_sym"]
            tau_sym = self.params["tau_sym"]
            I_asym = self.params["I_asym"]
            I_sym = self.params["I_sym"]
            I_sev = self.params["I_sev"]
            res.append(
                mu_sym[int(t)] * I_asym
                + mu_sym[int(t)] * I_sev
                - (
                    gamma_sym[int(t)] * I_sym
                    + mu_sev[int(t)] * I_sym
                    + tau_sym[int(t)] * I_sym
                )
            )
        elif "I2" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_dI_sevdt(self, t) -> np.ndarray:
        """
        Set up equation for Isev (severe infectious)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Isev for current iteration
        """
        res = []
        # I3 doesn't have to be checked because I_sev is only called if I3 and Q3 are in simulation_type
        mu_sev = self.params["mu_sev"]
        gamma_sev_r = self.params["gamma_sev_r"]
        gamma_sev_d = self.params["gamma_sev_d"]
        mu_sym = self.params["mu_sym"]
        tau_sev = self.params["tau_sev"]
        I_asym = self.params["I_asym"]
        I_sym = self.params["I_sym"]
        I_sev = self.params["I_sev"]
        Q_sev = self.params["Q_sev"]
        res.append(
            mu_sev[int(t)] * I_asym
            + mu_sev[int(t)] * I_sym
            - 1
            * (
                [
                    np.array(
                        [
                            (
                                (1 - self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k]))
                                * gamma_sev_r[int(t), j, k]
                                + self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                                * gamma_sev_d[int(t), j, k]
                            )
                            * I_sev[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
                + mu_sym[int(t)] * I_sev
                + tau_sev[int(t)] * I_sev
            )
        )

        return np.array(res).sum(axis=0)

    def _build_dQdt(self, t) -> Tuple:
        """
        Wrap equation for class Q (quarantined)
        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        Tuple
            Equation for class Q based on simulation_type

        """
        cls = self.simulation_type.split()
        if "Q3" in cls and "I3" in cls:
            return (
                self._build_dQ_asymdt(t),
                self._build_dQ_symdt(t),
                self._build_dQ_sevdt(t),
            )
        elif "Q2" in cls:
            return self._build_dQ_asymdt(t), self._build_dQ_symdt(t)
        elif "Q" in cls:
            pass

    def _build_dQ_asymdt(self, t) -> np.ndarray:
        """
        Set up equation for Qasym (asymptomatic quarantined)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Qasym for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "E2" in cls:
            epsilon = self.params["epsilon"]
            Etr = self.params["E_tr"]
            res.append(epsilon[int(t)] * Etr)
        if "I3" in cls:
            tau_asym = self.params["tau_asym"]
            I_asym = self.params["I_asym"]
            res.append(tau_asym[int(t)] * I_asym)
        elif "I2" in cls:
            pass

        if "Q3" in cls:
            gamma_asym = self.params["gamma_asym"]
            mu_sym = self.params["mu_sym"]
            mu_sev = self.params["mu_sev"]
            Q_asym = self.params["Q_asym"]
            res.append(
                -1
                * (
                    gamma_asym[int(t)] * Q_asym
                    + mu_sym[int(t)] * Q_asym
                    + mu_sev[int(t)] * Q_asym
                )
            )
        elif "Q2" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_dQ_symdt(self, t) -> np.ndarray:
        """
        Set up equation for Qsym (symptomatic quarantined)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Qsym for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "I3" in cls:
            tau_sym = self.params["tau_sym"]
            I_sym = self.params["I_sym"]
            res.append(tau_sym[int(t)] * I_sym)
        elif "I2" in cls:
            pass

        if "Q3" in cls:
            mu_sym = self.params["mu_sym"]
            gamma_sym = self.params["gamma_sym"]
            mu_sev = self.params["mu_sev"]
            Q_asym = self.params["Q_asym"]
            Q_sev = self.params["Q_sev"]
            Q_sym = self.params["Q_sym"]
            res.append(
                mu_sym[int(t)] * Q_asym
                + mu_sym[int(t)] * Q_sev
                - (gamma_sym[int(t)] * Q_sym + mu_sev[int(t)] * Q_sym)
            )
        elif "Q2" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_dQ_sevdt(self, t) -> np.ndarray:
        """
        Set up equation for Qsev (severe quarantined)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of Qsev for current iteration
        """
        res = []
        # Q3 doesn't have to be checked because Q_sev is only called if I3 and Q3 are in simulation_type
        tau_sev = self.params["tau_sev"]
        I_sev = self.params["I_sev"]
        res.append(tau_sev[int(t)] * I_sev)

        mu_sev = self.params["mu_sev"]
        mu_sym = self.params["mu_sym"]
        gamma_sev_r = self.params["gamma_sev_r"]
        gamma_sev_d = self.params["gamma_sev_d"]
        Q_asym = self.params["Q_asym"]
        Q_sym = self.params["Q_sym"]
        Q_sev = self.params["Q_sev"]
        I_sev = self.params["I_sev"]
        res.append(
            mu_sev[int(t)] * Q_asym
            + mu_sev[int(t)] * Q_sym
            - 1
            * (
                [
                    np.array(
                        [
                            (
                                (1 - self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k]))
                                * gamma_sev_r[int(t), j, k]
                                + self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                                * gamma_sev_d[int(t), j, k]
                            )
                            * Q_sev[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
                + mu_sym[int(t)] * Q_sev
            )
        )

        return np.array(res).sum(axis=0)

    def _build_dRdt(self, t) -> np.ndarray:
        """
        Set up equation for R (recovered)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of R for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "I3" in cls:
            gamma_asym = self.params["gamma_asym"]
            gamma_sym = self.params["gamma_sym"]
            gamma_sev_r = self.params["gamma_sev_r"]
            I_asym = self.params["I_asym"]
            I_sym = self.params["I_sym"]
            I_sev = self.params["I_sev"]
            Q_sev = self.params["Q_sev"]
            res.append(
                gamma_asym[int(t)] * I_asym
                + gamma_sym[int(t)] * I_sym
                + (
                    [
                        np.array(
                            [
                                1 - self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
                * gamma_sev_r[int(t)]
                * I_sev
            )
        elif "I2" in cls:
            pass
        elif "I" in cls:
            pass

        if "Q3" in cls:
            gamma_asym = self.params["gamma_asym"]
            gamma_sym = self.params["gamma_sym"]
            gamma_sev_r = self.params["gamma_sev_r"]
            Q_asym = self.params["Q_asym"]
            Q_sym = self.params["Q_sym"]
            Q_sev = self.params["Q_sev"]
            I_sev = self.params["I_sev"]
            res.append(
                gamma_asym[int(t)] * Q_asym
                + gamma_sym[int(t)] * Q_sym
                + (
                    [
                        np.array(
                            [
                                1 - self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                                for k in range(self.K)
                            ]
                        )
                        for j in range(self.J)
                    ]
                )
                * gamma_sev_r[int(t)]
                * Q_sev
            )
        elif "Q2" in cls:
            pass
        elif "Q" in cls:
            pass

        # method _build_dR_dt is only called if R is in simulation_type so this has to be executed anways
        rho_rec = self.params["rho_rec"]
        R = self.params["R"]
        res.append(-1 * rho_rec[int(t)] * R)

        return np.array(res).sum(axis=0)

    def _build_dDdt(self, t) -> np.ndarray:
        """
        Set up equation for D (dead)

        Parameters
        ----------
        t : int
            Current timestep

        Returns
        -------
        np.ndarray
            Calculated values of equation of D for current iteration
        """
        res = []
        cls = self.simulation_type.split()
        if "I3" in cls:
            gamma_sev_d = self.params["gamma_sev_d"]
            I_sev = self.params["I_sev"]
            Q_sev = self.params["Q_sev"]
            res.append(
                [
                    np.array(
                        [
                            self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                            * gamma_sev_d[int(t), j, k]
                            * I_sev[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
            )
        elif "I2" in cls:
            pass
        elif "I" in cls:
            pass

        if "Q3" in cls:
            gamma_sev_d = self.params["gamma_sev_d"]
            I_sev = self.params["I_sev"]
            Q_sev = self.params["Q_sev"]
            res.append(
                [
                    np.array(
                        [
                            self._calc_sigma(t, j, k, I_sev[j, k], Q_sev[j, k])
                            * gamma_sev_d[int(t), j, k]
                            * Q_sev[j, k]
                            for k in range(self.K)
                        ]
                    )
                    for j in range(self.J)
                ]
            )
        elif "Q2" in cls:
            pass
        elif "Q" in cls:
            pass

        return np.array(res).sum(axis=0)

    def _build_ode_system(self, t, params: np.ndarray) -> np.ndarray:
        """
        builds an ODE system for a given simulation type.
        E.g. simulation_type = "SI" -> [dSdt = ..., dIdt = ...]

        Parameters
        ----------
        t : np.ndarray
            timesteps of simulation - is not used actively but is necessary for the call of solve_ivp
        params : np.ndarray
            parameters for the simulation

        Returns
        -------
        np.ndarray
            1-Dim-Array of Ordinary differential equations that build a system
        """
        # Simulation doesn't do useful things if self.params won't change
        # -> so this reshape is necessary to set self.params to the calculated data from the previous iteration
        # so the current iteration can use these calculated data to use them in the next calculation
        if self.simulation_type == "S I":
            pass
        elif self.simulation_type == "S E I R":
            pass
        if self.simulation_type == "M V S E2 I3 Q3 R D":
            (
                self.params["M"],
                self.params["V"],
                self.params["S"],
                self.params["E_tr"],
                self.params["E_nt"],
                self.params["I_asym"],
                self.params["I_sym"],
                self.params["I_sev"],
                self.params["Q_asym"],
                self.params["Q_sym"],
                self.params["Q_sev"],
                self.params["R"],
                self.params["D"],
            ) = params.reshape((self.param_count, self.J, self.K))

        result = []
        if "M" in self.simulation_type:
            result.append(self._build_dMdt(t))
        else:
            result.append(np.zeros((self.J, self.K)))

        if "V" in self.simulation_type:
            result.append(self._build_dVdt(t))
        else:
            result.append(np.zeros((self.J, self.K)))

        if "S" in self.simulation_type:
            result.append(self._build_dSdt(t))
        else:
            result.append(np.zeros((self.J, self.K)))

        if "E" in self.simulation_type:
            e = self._build_dEdt(t)
            for sub_types in e:
                result.append(sub_types)
        else:
            result.append(np.zeros((self.J, self.K)))

        if "I" in self.simulation_type:
            i = self._build_dIdt(t)
            for sub_types in i:
                result.append(sub_types)
        else:
            result.append(np.zeros((self.J, self.K)))

        if "Q" in self.simulation_type:
            q = self._build_dQdt(t)
            for sub_types in q:
                result.append(sub_types)
        else:
            result.append(np.zeros((self.J, self.K)))

        if "R" in self.simulation_type:
            result.append(self._build_dRdt(t))
        else:
            result.append(np.zeros((self.J, self.K)))

        if "D" in self.simulation_type:
            result.append(self._build_dDdt(t))
        else:
            result.append(np.zeros((self.J, self.K)))

        return np.array(result).ravel()

    def _run_ode_system(self, params) -> dict:
        """
        Creates ODE system for epidemiological model

        Parameters
        ----------
        params : dict
            Parameters for the simulation

        Returns
        -------
        dict
            result of simulation
        """
        t = np.arange(0, params["t"], 1)
        return self._simulate_RK45(t, params)

    def _calc_sigma(self, t, j, k, I_sev_jk, Q_sev_jk) -> np.float64:
        """
        Calculates value of sigma for specific situation dependent on available hospital beds

        Parameters
        ----------
        t : int
            Current timestep
        j : int
            index of current district
        k : int
            index of current group
        I_sev_jk : np.float64
            current value of I_sev for district j in group k
        Q_sev_jk : np.float64
            current value of Q_sev for district j in group k

        Returns
        -------
        np.float64
            calculated value of sigma for current iteration
        """
        N = self.params["N"]
        N_total = np.sum(N, axis=1)
        Beds = self.params["Beds"]
        sigma = self.params["sigma"]
        if (I_sev_jk + Q_sev_jk) * N[j, k] <= (N[j, k] / N_total[j]) * Beds[j]:
            return sigma[int(t), j, k]
        else:
            return (
                sigma[int(t), j, k] * (N[j, k] / N_total[j]) * Beds[j]
                + (I_sev_jk + Q_sev_jk) * N[j, k]
                - (N[j, k] / N_total[j]) * Beds[j]
            ) / ((I_sev_jk + Q_sev_jk) * N[j, k])

    def _simulate_RK45(self, t, params) -> np.ndarray:
        """
        Use solve_ivp with method 'RK45' of scipy.integrate

        Parameters
        ----------
        t : np.ndarray
            timesteps of simulation
        params : dict
            parameters for the simulation

        Returns
        -------
        np.ndarray
            solution of ODE system solved with scipy.solve_ivp
        """
        return self._simulate_ivp(t, params, scipy.integrate.RK45)

    def _simulate_RK23(self, t, params) -> np.ndarray:
        """
        Use solve_ivp with method 'RK23' of scipy.integrate

        Parameters
        ----------
        t : np.ndarray
            timesteps of simulation
        params : dict
            parameters for the simulation

        Returns
        -------
        np.ndarray
            solution of ODE system solved with scipy.solve_ivp
        """
        return self._simulate_ivp(t, params, scipy.integrate.RK23)

    def _simulate_DOP853(self, t, params) -> np.ndarray:
        """
        Use solve_ivp with method 'DOP853' of scipy.integrate

        Parameters
        ----------
        t : np.ndarray
            timesteps of simulation
        params : dict
            parameters for the simulation

        Returns
        -------
        np.ndarray
            solution of ODE system solved with scipy.solve_ivp
        """
        return self._simulate_ivp(t, params, scipy.integrate.DOP853)

    def _prepare_y0(self) -> list:
        """
        Prepare start values for the current iteration
        Returns
        -------
        list
            Start values for current iteration
        """
        if self.simulation_type == "S I":
            return [self.params["S"], self.params["I"]]
        if self.simulation_type == "S E I R":
            return [
                self.params["S"],
                self.params["E"],
                self.params["I"],
                self.params["R"],
            ]
        if self.simulation_type == "M V S E2 I3 Q3 R D":
            return [
                self.params["M"],
                self.params["V"],
                self.params["S"],
                self.params["E_tr"],
                self.params["E_nt"],
                self.params["I_asym"],
                self.params["I_sym"],
                self.params["I_sev"],
                self.params["Q_asym"],
                self.params["Q_sym"],
                self.params["Q_sev"],
                self.params["R"],
                self.params["D"],
            ]

    def _simulate_ivp(self, t, params, method) -> np.ndarray:
        """
        Solve ODE system with solve_ivp

        Parameters
        t : np.ndarray
            timesteps of simulation
        params : dict
            parameters for the simulation

        Returns
        -------
        np.ndarray
            solution of ODE system solved with scipy.solve_ivp
        """
        # TODO find better way
        for key in params:
            if params[key] is not None:
                if key in [ "beta_asym", "beta_sym", "beta_sev"]:
                    self.params[key] = np.ones((len(t), self.J, self.K, self.K)) * np.array(params[key])
                elif key in ["M", "V", "R", "S", "E", "E_tr", "E_nt",
                             "I", "I_asym", "I_sym", "I_sev", "Q", "Q_asym", "Q_sym", "Q_sev",
                             "D"]:
                    self.params[key] = np.ones((self.J, self.K)) * np.array(params[key])
                elif key not in ["K", "J", "N", "t", "Beds"]:
                    self.params[key] = np.ones((len(t), self.J, self.K)) * np.array(params[key])

        sol = solve_ivp(
            fun=self._build_ode_system,
            t_span=[t[0], t[-1]],
            t_eval=t,
            y0=np.array(self._prepare_y0()).ravel(),
            method=method,
        )

        result = [
            np.array(
                [sol.y[:, i].reshape((self.param_count, self.J, self.K))[j] for i in range(len(t))]
            )
            for j in range(13)
        ]

        return result

    def _simulate_odeint(self, params, t) -> np.ndarray:
        """
        Solve ODE system with odeint

        Parameters
        t : np.ndarray
            timesteps of simulation
        params : dict
            parameters for the simulation

        Returns
        -------
        np.ndarray
            solution of ODE system solved with scipy.solve_ivp
        """
        sol = odeint(
            func=self._build_ode_system, t=t, y0=np.array(self._prepare_y0()).ravel(), tfirst=True,
        )

        result = [
            np.array(
                [sol[i, :].reshape((self.param_count, self.J, self.K))[j] for i in range(len(t))]
            )
            for j in range(13)
        ]

        return result
