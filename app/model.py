import numpy as np

from .controller import Controller
from .simulator import Simulation_Algorithm, Simulator
from .view import View


class Model:
    def __init__(
        self,
        params: dict = None,
        fill_missing_values: bool = True,
        load_base_parameter: bool = True,
        default_values_path="data/default_values.json",
        default_domains_path="data/default_domains.json",
    ) -> None:
        """
        Epidemiological model for the coronavirus pandemic based on the SI Model. 
        Currently supported modelations are ["S I", "M V S E2 I3 Q3 R D"]. 
        For a more detailed description see presentation.ipynb and the Simulator class.

        Optional Parameters:
         params
         default_values_path
         default_domains_path
        """
        self.controller = Controller(
            model=self,
            params=params,
            fill_missing_values=fill_missing_values,
            load_base_parameter=load_base_parameter,
            default_values_path=default_values_path,
            default_domains_path=default_domains_path,
        )
        self.get_params = self.controller.get_params
        self.update = self.controller.update
        self.simulator = Simulator(self)
        self.view = View(self)
        self.plot = self.view.plot

    def detect_simulation_type(self, params: dict) -> str:
        """
        Detects from a dict of parameters which epidemiological simulation can be run.
        E.g. if only beta is given, only the SI model can be run 
        For supported simulation types see class description.

        Returns
        -------
        str
            simulation_type (e.g. "S I", "S E I R", "M V S E2 I3 Q3 R D")
        """
        # Not None keys
        keys = [
            key
            for key in params.keys()
            if isinstance(params[key], np.ndarray) or isinstance(params[key], list)
        ]

        sim_type = ""

        # M
        if "rho_mat" in keys:
            sim_type += "M"

        # V
        if "rho_vac" in keys:
            sim_type += " V"

        # S
        if "beta_asym" in keys or "beta_sym" in keys or "beta_sev" in keys:
            sim_type += " S"
        elif "beta" in keys:
            sim_type += " S"
        else:
            raise ValueError("Parameters must contain infection rate beta.")

        # E
        if "epsilon" in keys:
            if "psi" in keys:
                sim_type += " E2"
            else:
                sim_type += " E"

        # I
        if "mu_sym" in keys:
            if "mu_sev" in keys:
                sim_type += " I3"
            else:
                sim_type += " I2"
        else:
            sim_type += " I"

        # Q
        if "tau_sym" in keys and "tau_asym" in keys:
            if "tau_sev" in keys:
                sim_type += " Q3"
            else:
                sim_type += " Q2"
        elif "tau" in keys:
            sim_type += " Q"

        # R
        if "gamma_sym" in keys:
            sim_type += " R"

        # D
        if "sigma" in keys:
            sim_type += " D"

        if sim_type in self.simulator.supported_sim_types:
            return sim_type
        else:
            raise ValueError(
                f"Simulation type {sim_type} not supported. "
                "For supported simulation types see Controller.supported_sim_types."
            )

    def get_simulation_type(self) -> str:
        return self.simulator.simulation_type

    def translate_simulation_type(
        self, simulation_type: str = None, return_classes: bool = True, return_greeks: bool = False
    ) -> list:
        """
        Translates simulation type from detect_simulation_type into a list of 
        identifiers for epidemiological classes used in params. 
        Important for classes such as I3, so convert into I_asym, I_sym, I_sev.

        If return_classes is set to False, no classes will be returned.
        If return_greeks is set to True, hyperparameter will be returned.

        Parameters
        ----------
        simulation_type : str, optional
            simulation type to be translated, 
            by default the current simulation type is read from the simulator

        Returns
        -------
        list
            identifiers for params dict
        
        Examples
        -------
        >>> translate_simulation_type("E2")
        >>> ['E_nt', 'E_tr']
        """
        params = self.get_params()

        # No simulation type supplied
        if not simulation_type:
            simulation_type = self.get_simulation_type()
            # If simulation has not been run before
            if not simulation_type:
                simulation_type = self.detect_simulation_type(params).split()
            else:
                simulation_type = simulation_type.split()
        else:
            simulation_type = simulation_type.split()
        classes = []
        for letter in simulation_type:
            if letter == "M":
                classes.append("M")
            elif letter == "V":
                classes.append("V")
            elif letter == "S":
                classes.append("S")
            elif letter == "E":
                classes.append("E")
            elif letter == "E2":
                classes.append("E_nt")
                classes.append("E_tr")
            elif letter == "I":
                classes.append("I")
            elif letter == "I2":
                classes.append("I_asym")
                classes.append("I_sym")
            elif letter == "I3":
                classes.append("I_asym")
                classes.append("I_sym")
                classes.append("I_sev")
            elif letter == "Q":
                classes.append("Q")
            elif letter == "Q2":
                classes.append("Q_asym")
                classes.append("Q_sym")
            elif letter == "Q3":
                classes.append("Q_asym")
                classes.append("Q_sym")
                classes.append("Q_sev")
            elif letter == "R":
                classes.append("R")
            elif letter == "D":
                classes.append("D")

        if return_classes and not return_greeks:
            return classes

        greeks = [
            "beta_asym",
            "beta_sym",
            "beta_sev",
            "gamma_asym",
            "gamma_sym",
            "gamma_sev",
            "gamma_sev_r",
            "gamma_sev_d",
            "epsilon",
            "mu_sym",
            "mu_sev",
            "nu",
            "rho_mat",
            "rho_vac",
            "rho_rec",
            "sigma",
            "tau_asym",
            "tau_sym",
            "tau_sev",
            "psi",
        ]
        # Not None greeks
        greeks = [
            key
            for key in params.keys()
            if isinstance(params[key], np.ndarray)
            or isinstance(params[key], list)
            and key in greeks
        ]

        if return_classes and return_greeks:
            return classes + greeks
        elif return_greeks:
            return greeks

    def reset_parameters(self):
        """
        Resets the internal parameter list to an empty list. 
        If run(fill_missing_values=True) is called after, the controller will use default values
        """
        self.controller.reset()

    def run(
        self, simulation_algorithm: Simulation_Algorithm = Simulation_Algorithm.SOLVE_IVP
    ) -> dict:
        """
        Retrieves parameter, runs the MVRSEIQRI simulation and updates the view.
        This is equivalent to simulating one timestep.
        """

        # call simulator with simulation type and current parameters
        params = self.controller.get_full_params(use_original_classes_data=True)
        simulation_type = self.detect_simulation_type(params)
        classes_keys = self.translate_simulation_type(simulation_type)
        res = self.simulator.run(
            params=params,
            simulation_type=simulation_type,
            simulation_algorithm=simulation_algorithm,
        )
        simulated_params = dict(zip(classes_keys, res))
        self.controller.update(params=simulated_params, fill_missing_values=True, reset=False)
        params = self.controller.get_params()

        return simulated_params

    def run_app(self):
        return self.view.run_app()
