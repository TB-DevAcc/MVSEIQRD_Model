class Model:
    def __init__(self) -> None:
        # TODO implement
        ...

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
        # TODO implement
        ...

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
        Runs the ISEIQRI simulation and updates the view
        """
        params = self._retrieve_parameter()
        # TODO make sure entities and age groups are correct in every parameter
        # TODO make sure params[J] (entities number) and params[K] (age group number) are available

        # TODO call simulator with simulation type
        ...

        # TODO update the view
        self._update_view(params)
