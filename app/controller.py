import json


class Controller:
    """
    Controls parameters and handles information from view.
    
    Parameters
    ----------
    # Population classes
    self._M = None  # Maternity-derived immunity

    self._V = None  # Vaccinated

    self._R = None  # Recovered

    self._S = None  # Susceptible

    self._E = None  # Exposed
    self._E_tr = None
    self._E_nt = None

    self._I = None  # Infectious
    self._I_asym = None
    self._I_sym = None
    self._I_sev = None

    self._Q = None  # Quarantined
    self._Q_asym = None
    self._Q_sym = None
    self._Q_sev = None

    self._D = None  # Dead

    # Environment
    self._N = None  # Population
    self._K = None  # Subgroups
    self._basic_reprod_num = None  # Basic Reproduction Number, don't confuse with R_0
    self._Beds = None  # Number of ICU beds per LK

    # Disease parameter
    self._beta_asym = None  # Infection rate
    self._beta_sym = None
    self._beta_sev = None

    self._gamma_asym = None  # Average infectious period
    self._gamma_sym = None
    self._gamma_sev = None
    self._gamma_sev_r = None
    self._gamma_sev_d = None

    self._epsilon = None  # Average latent period

    self._mu = None  # Symptom changing rate
    self._mu_asym = None
    self._mu_sym = None
    self._mu_sev = None

    self._nu = None  # Vaccionation rate

    self._rho = None  # Average immunity period
    self._rho_mat = None
    self._rho_vac = None
    self._rho_rec = None

    self._sigma = None  # Deathrate

    self._tau = None  # Quarantining rate

    self._psi = None  # Tracing app usage rate
    """

    def __init__(self, default_values_path="/data/default_values.json"):
        # Population classes
        self._M = None  # Maternity-derived immunity

        self._V = None  # Vaccinated

        self._R = None  # Recovered

        self._S = None  # Susceptible

        self._E = None  # Exposed
        self._E_tr = None
        self._E_nt = None

        self._I = None  # Infectious
        self._I_asym = None
        self._I_sym = None
        self._I_sev = None

        self._Q = None  # Quarantined
        self._Q_asym = None
        self._Q_sym = None
        self._Q_sev = None

        self._D = None  # Dead

        # Environment
        self._N = None  # Population
        self._K = None  # Subgroups
        self._basic_reprod_num = None  # Basic Reproduction Number, don't confuse with R_0
        self._Beds = None  # Number of ICU beds per LK

        # Disease parameter
        self._beta_asym = None  # Infection rate
        self._beta_sym = None
        self._beta_sev = None

        self._gamma_asym = None  # Average infectious period
        self._gamma_sym = None
        self._gamma_sev = None
        self._gamma_sev_r = None
        self._gamma_sev_d = None

        self._epsilon = None  # Average latent period

        self._mu = None  # Symptom changing rate
        self._mu_asym = None
        self._mu_sym = None
        self._mu_sev = None

        self._nu = None  # Vaccionation rate

        self._rho = None  # Average immunity period
        self._rho_mat = None
        self._rho_vac = None
        self._rho_rec = None

        self._sigma = None  # Deathrate

        self._tau = None  # Quarantining rate

        self._psi = None  # Tracing app usage rate

        self.default_values = self._load_default_values(default_values_path)

    def _load_default_start_values(self, default_values_path) -> dict:
        try:
            file = json.loads(default_values_path)[0]
        except FileExistsError as fee:
            raise fee
        if isinstance(file, dict):
            return file
        else:
            raise FileNotFoundError(f"{default_values_path} does not contain any values.")

    def valid_attribution(self, key, value) -> bool:
        """
        Checks if parameter value is correctly attributed within its defined domain

        Parameters
        ----------
        key : immutable data type / dictionary key
            dictionary key of class attribute
        value : Any
            value of class attribute described by key

        Returns
        -------
        bool
            Returns True if the value is in its defined domain, False otherwise
        """
        try:
            self.key
        except AttributeError as ae:
            # shouldn't occur, check attribute config file
            raise ae(
                "You have checked for an attribute key that does not exist."
                "This should generally not occur unless new attributes have been"
                "defined without adding them to the attribute config file."
                "Please check your input and run again."
            )

        # TODO Define domain ranges for values and classes, perhaps through JSON file?
        ...

        return False

    def _check_params(self) -> None:
        """
        Checks if parameter are still in their defined domain

        Raises
        -------
        AssertionError
            Raises AssertionError of attribute if valid_attribution returns False for its domain
        """
        for attr in self.__dict__:
            if not self.valid_attribution(attr, self.attr):
                raise AttributeError(
                    f"Invalid attribute, {attr} can not be of value {self.attr}."
                    "Check domain definition."
                )

    def initialize_parameters(self, params: dict = None) -> None:
        """
        Initializes all parameters either by default or by reading in data 
        (usually prepared with the data handler)
        """
        # TODO Initialize all parameters with their start _0 value;
        # make sure types are clear first under valid_attribution and then initialize within bounds

        for key in params:
            if params[key]:
                self.set_value(key=key, value=params[key])
            else:
                try:
                    self.set_value(key=key, value=self.default_start_values[key])
                except AttributeError as ae:
                    raise ae(
                        f"The attribute {key} has no default starting value. Please supply "
                        "a value or specifify the value in the starting values config file"
                    )

    def set_value(self, key, value) -> None:
        """
        Sets a class attribute's value

        Parameters
        ----------
        key : immutable data type / dictionary key
            dictionary key of class attribute
        value : Any
            value of class attribute described by key
        """
        # TODO setter method for class values;
        # make sure to handle domain defintion checking e.g. through running valid_attribution first

    def get_values(self) -> dict:
        """
        Prepares class attributes for retrieval and makes sure values are valid
        """
        # Alternatively save class attributes to suitable storage (file, database, etc)
        # TODO Prepare attributes for retrieval
        # Return attributes compact and in a fixed format
