import numpy as np

from .controller import Controller
from .simulator import Simulator
from .view import View


class Model:
    def __init__(
        self,
        params: dict = None,
        fill_missing_values: bool = True,
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
            default_values_path=default_values_path, default_domains_path=default_domains_path
        )
        self.get_params = self.controller.get_params
        self.simulator = Simulator()
        self.view = View(self)
        self.plot = self.view.plot

        # TODO load data from DataHandler and put them into params
        # self.data_handler = DataHandler()
        # base_data = self.data_handler.get_base_values()
        # if len(base_data) > 0:
        #     params = {"N": [], "Beds": []}
        #     for key, value in self.data_handler.get_base_values().items():
        #         params["N"].append(value["N"])
        #         params["Beds"].append(value["B"])
        #
        # self.controller.set_params(params, False)

        self._update_params(params, fill_missing_values, reset=True)

    def _update_view(self, params) -> None:
        """
        Updates the view if new data is available
        """
        # TODO implement
        ...

    def _update_params(self, params, fill_missing_values, reset=False) -> None:
        """
        Updates the controller if new data/parameters is/are available. 
        Sets self.controller._params to current parameter dict.
        """
        if reset:
            self.controller.reset()

        if fill_missing_values:
            # build complete parameter set with controller
            # add default values to given params dict
            self.controller.initialize_parameters(params)
        else:
            # add given params to already existing parameter dict
            self.controller.check_params(params)
            self.controller.set_params(params)

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

    def translate_simulation_type(self, simulation_type: str = None) -> list:
        """
        Translates simulation type from detect_simulation_type into a list of 
        identifiers for epidemiological classes used in params. 
        Important for classes such as I3, so convert into I_asym, I_sym, I_sev.

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
        # No simulation type supplied
        if not simulation_type:
            simulation_type = self.get_simulation_type()
            # If simulation has not been run before
            if not simulation_type:
                simulation_type = self.detect_simulation_type(self.get_params()).split()
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
        return classes

    def reset_parameters(self):
        """
        Resets the internal parameter list to an empty list. 
        If run(fill_missing_values=True) is called after, the controller will use default values
        """
        self.controller.reset()

    def run(self) -> dict:
        """
        Retrieves parameter, runs the MVRSEIQRI simulation and updates the view.
        This is equivalent to simulating one timestep.
        """

        # call simulator with simulation type and current parameters
        params = self.controller.get_params()
        simulation_type = self.detect_simulation_type(params)
        retParams = {}
        (
            retParams["M"],
            retParams["V"],
            retParams["S"],
            retParams["E_tr"],
            retParams["E_nt"],
            retParams["I_asym"],
            retParams["I_sym"],
            retParams["I_sev"],
            retParams["Q_asym"],
            retParams["Q_sym"],
            retParams["Q_sev"],
            retParams["R"],
            retParams["D"],
        ) = self.simulator.run(params=params, simulation_type=simulation_type)
        self._update_params(params=retParams, fill_missing_values=True, reset=False)
        params = self.controller.get_params()

        # update the view
        self._update_view(params)
        return params

    def run_app(self):
        return self.view.run_app()
