from controller import Controller
from simulator import Simulator
from view import View


class Model:
    def __init__(
        self,
        params: dict = None,
        default_values_path="/data/default_values.json",
        default_domains_path="/data/default_domains.json",
    ) -> None:
        """
        Optional Parameters:
         params
         default_values_path
         default_domains_path
        """
        if params:
            self._params = params
        else:
            self._params = {}

        self.simulation_type = None
        self.controller = Controller(
            default_values_path=default_values_path, default_domains_path=default_domains_path
        )
        self.simulator = Simulator()
        self.view = View()

    def _update_view(self) -> None:
        """
        Updates the view if new data is available
        """
        # TODO implement
        ...

    def _detect_simulation_type(self, params: dict) -> str:
        """
        Detects from a dict of parameters which epidemiological simulation can be run.
        E.g. if only beta and S are given, only the SI model can be run 

        Returns
        -------
        str
            simulation_type (e.g. "SI", "SEIR", "ISEIQR")
        """
        # TODO implement
        return "MVRSEIQRI"

    def reset_parameters(self):
        """
        Resets the internal parameter list to an empty list. 
        If run(fill_missing_values=True) is called after, the controller will use default values
        """
        self._params = {}

    def run(self, fill_missing_values: bool = False) -> None:
        """
        Retrieves parameter, runs the MVRSEIQRI simulation and updates the view.
        This is equivalent to simulating one timestep.
        """
        # build complete parameter set with controller
        if fill_missing_values:
            params = self.controller.initialize_parameters(self._params)
        else:
            params = self.controller.check_params(self._params)

        # call simulator with simulation type
        sim_type = self._detect_simulation_type(params)
        self._params = self.simulator.run(params=params, simulation_type=sim_type)

        # update the view
        self._update_view(self._params)
