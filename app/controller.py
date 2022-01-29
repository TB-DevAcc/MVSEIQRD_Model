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

    def __init__(
        self,
        default_values_path="/data/default_values.json",
        default_domains_path="/data/default_domains.json",
    ):
        """
        SUPPORTED PARAMETERS
        --------------------

        # Population classes
        M in [0, 1]                   # Maternity-derived immunity

        V in [0, 1]                   # Vaccinated

        R in [0, 1]                   # Recovered

        S in [0, 1]                   # Susceptible

        E in [0, 1]                   # Exposed
        E_tr in [0, 1]                  
        E_nt in [0, 1]                  

        I in [0, 1]                   # Infectious
        I_asym in [0, 1]                    
        I_sym in [0, 1]                 
        I_sev in [0, 1]                 

        Q in [0, 1]                   # Quarantined
        Q_asym in [0, 1]                    
        Q_sym in [0, 1]                 
        Q_sev in [0, 1]                 

        D in [0, 1]                   # Dead

        # Environment
        N in [0, 1]                   # Population
        K in [0, 1]                   # Subgroups
        basic_reprod_num in [0, 1]    # Basic Reproduction Number, don't confuse with R_0
        Beds in [0, 1]                # Number of ICU beds per LK

        # Disease parameter
        beta_asym in [0, 1]           # Infection rate
        beta_sym in [0, 1]                  
        beta_sev in [0, 1]                  

        gamma_asym in [0, 1]          # Average infectious period
        gamma_sym in [0, 1]                 
        gamma_sev in [0, 1]                 
        gamma_sev_r in [0, 1]                   
        gamma_sev_d in [0, 1]                   

        epsilon in [0, 1]             # Average latent period

        mu in [0, 1]                  # Symptom changing rate
        mu_asym in [0, 1]                   
        mu_sym in [0, 1]                    
        mu_sev in [0, 1]                    

        nu in [0, 1]                  # Vaccionation rate

        rho in [0, 1]                 # Average immunity period
        rho_mat in [0, 1]                   
        rho_vac in [0, 1]                   
        rho_rec in [0, 1]                   

        sigma in [0, 1]               # Deathrate

        tau in [0, 1]                 # Quarantining rate

        psi in [0, 1]                 # Tracing app usage rate

        J in [0, 1]                   # geographic locations
        K in [0, 1]                   # age groups
        """
        self.params = {
            "M": None,
            "V": None,
            "R": None,
            "S": None,
            "E": None,
            "E_tr": None,
            "E_nt": None,
            "I": None,
            "I_asym": None,
            "I_sym": None,
            "I_sev": None,
            "Q": None,
            "Q_asym": None,
            "Q_sym": None,
            "Q_sev": None,
            "D": None,
            "N": None,
            "K": None,
            "basic_reprod_num": None,
            "Beds": None,
            "beta_asym": None,
            "beta_sym": None,
            "beta_sev": None,
            "gamma_asym": None,
            "gamma_sym": None,
            "gamma_sev": None,
            "gamma_sev_r": None,
            "gamma_sev_d": None,
            "epsilon": None,
            "mu": None,
            "mu_asym": None,
            "mu_sym": None,
            "mu_sev": None,
            "nu": None,
            "rho": None,
            "rho_mat": None,
            "rho_vac": None,
            "rho_rec": None,
            "sigma": None,
            "tau": None,
            "psi": None,
            "J": None,
            "K": None,
        }
        self.default_values = self._load_default_values(default_values_path)
        self.default_domains = self._load_default_domains(default_domains_path)

    def reset(self):
        """
        Resets the current controllers paramter dict to None
        """
        for key in self.params:
            self.params[key] = None

    def _load_default_values(self, default_values_path) -> dict:
        try:
            file = json.loads(default_values_path)[0]
        except FileExistsError as fee:
            raise fee
        if isinstance(file, dict):
            return file
        else:
            raise FileNotFoundError(f"{default_values_path} does not contain any values.")

    def _load_default_domains(self, default_domains_path) -> dict:
        try:
            file = json.loads(default_domains_path)[0]
        except FileExistsError as fee:
            raise fee
        if isinstance(file, dict):
            return file
        else:
            raise FileNotFoundError(f"{default_domains_path} does not contain any values.")

    def valid_domain(self, key, value) -> bool:
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
            domain = self.default_domains[key]
        except AttributeError as ae:
            # shouldn't occur, check attribute config file
            raise ae(
                "You have checked for an attribute key that does not exist in the "
                "default parameter descriptions."
                "This should generally not occur unless new attributes have been"
                "defined without adding them to the attribute config files."
                "Please check your input and run again."
            )

        if len(domain) == 2:
            if domain[0] < value < domain[1]:
                return True
        else:
            raise ValueError(
                "Illegal domain configuration." f" Domain {domain} has more than two boundaries."
            )
        return False

    def check_params(self, params: dict) -> None:
        """
        Checks if parameter are still in their defined domain and fit other constraints.

        Raises
        -------
        AssertionError
            Raises AssertionError of attribute if valid_domain returns False for its domain
        ValueError
            Raises ValueError if attribute does not fit its constraints 
            e.g. J and K do not fit the dimensions of other parameters
        """
        # check domain
        for attr, val in params.items():
            if not self.valid_domain(attr, val):
                raise AttributeError(
                    f"Invalid attribute, {attr} can not be of value {self.attr}."
                    "Check domain definition."
                )

        # make sure params[J] (entities number) and params[K] (age group number) are available
        try:
            params["J"]
        except AttributeError as ae:
            # TODO logging info or raise Error
            self.params["J"] = self.default_values["J"]

        try:
            params["K"]
        except AttributeError as ae:
            # TODO logging info
            for key in params:
                if key not in ["K", "J", "beta"]:
                    self.params["K"] = len(params[key][0])
                    break

        # make sure entities and age groups are correct in every parameter
        for key in params:
            if key not in ["K", "J", "beta"]:
                if len(params[key][0]) != params["K"]:
                    raise ValueError(
                        f"K:{self._K} does not match the first dimension of the"
                        " parameters. If you intended to not subdivide the different classes,"
                        " choose K=1 and wrap your parameters in a list."
                    )
            elif key == "beta":
                # TODO check correct shape of beta
                pass

    def initialize_parameters(self, params: dict = None) -> dict:
        """
        Initializes all parameters either by default or by setting it with supplied params dict
        """
        # Initialize all parameters with their start _0 value;
        if not params:
            params = {}
        for key, val in self.default_values.items():
            if key in params:
                self.params[key] = params[key]
            else:
                self.params[key] = val

        # make sure types are clear first under valid_domain and then initialize within bounds
        self.check_params(self.params)

        return self.params

    def set_values(self, params: dict) -> None:
        """
        Sets a class attribute's value

        Parameters
        ----------
        params with
            key : immutable data type / dictionary key
                dictionary key of class attribute
            value : Any
                value of class attribute described by key
        """
        self.check_params(params)
        for key, val in params:
            self.params[key] = val

    def get_values(self, keys: list) -> dict:
        """
        Prepares class attributes for retrieval and makes sure values are valid
        
        Parameters
        ----------
            keys : list
                list of strings, that are keys for parameters
        """
        return {key: self.params[key] for key in keys}
