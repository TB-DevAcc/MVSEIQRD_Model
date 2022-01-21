from controller import Controller
from simulator import Simulator
from view import View


class Model:
    def __init__(self, **kwargs) -> None:
        """
        Optional Parameters:
         params
         J
         K
        """
        # TODO implement
        if "params" in kwargs:
            self.params = kwargs["params"]
        else:
            self.params = {}
        if "J" in kwargs:
            self.J = kwargs["J"]
        else:
            self.J = 1
        if "K" in kwargs:
            self.K = kwargs["K"]
        else:
            self.K = 1

        self.simulation_type = None

    def _update_view(self) -> None:
        """
        Updates the view if new data is available
        """
        # TODO implement
        ...

    def _retrieve_parameter(self) -> dict:
        """
        Retrieves the parameter for the simulation from the controller 

        Returns
        -------
        dict
            Parameter for the simulation
        """
        controller = Controller()
        controller.initialize_parameters(**self.params)

    def _detect_simulation_type(self) -> str:
        """
        Detects from a dict of parameters which epidemiological simulation can be run.
        E.g. if only beta and S are given, only the SI model can be run 

        Returns
        -------
        str
            simulation_type (e.g. "SI", "SEIR", "ISEIQR")
        """
        # TODO implement
        ...

    def run(self) -> None:
        """
        Retrieves parameter and runs the MVRSEIQRI simulation and updates the view
        """
        params = self._retrieve_parameter()
        # TODO make sure entities and age groups are correct in every parameter
        # TODO make sure params[J] (entities number) and params[K] (age group number) are available

        # TODO call simulator with simulation type
        ...

        # TODO update the view
        self._update_view(params)
