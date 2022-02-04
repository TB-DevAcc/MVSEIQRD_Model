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
        self.view = View()

        self._update_params(params, fill_missing_values, reset=True)

    def _update_view(self) -> None:
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
        keys = [key for key in params.keys if params[key]]

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
        if "gamma" in keys:
            sim_type += " R"

        # D
        if "sigma" in keys:
            sim_type += " D"

        if sim_type in self.simulator.supported_sim_types:
            return sim_type
        else:
            raise ValueError(
                f"Simulation type {sim_type} not supported."
                "For supported simulation types see Controller.supported_sim_types"
            )

    def reset_parameters(self):
        """
        Resets the internal parameter list to an empty list. 
        If run(fill_missing_values=True) is called after, the controller will use default values
        """
        self.controller.reset()

    def run(self) -> None:
        """
        Retrieves parameter, runs the MVRSEIQRI simulation and updates the view.
        This is equivalent to simulating one timestep.
        """

        # call simulator with simulation type and current parameters
        params = self.controller.get_params()
        simulation_type = self.detect_simulation_type(params)
        params = self.simulator.run(params=params, simulation_type=simulation_type)
        self._update_params(params=params, fill_missing_values=False, reset=False)

        # update the view
        self._update_view(self.controller.get_params())
