import json
from functools import reduce

import numpy as np

from .data_handler import DataHandler


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
        model,
        params: dict = None,
        fill_missing_values: bool = True,
        default_values_path="data/default_values.json",
        default_domains_path="data/default_domains.json",
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
        self.model = model
        self._params = {
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
            "J": None,
            "t": None,
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
            "mu_sym": None,
            "mu_sev": None,
            "nu": None,
            "rho_mat": None,
            "rho_vac": None,
            "rho_rec": None,
            "sigma": None,
            "tau_asym": None,
            "tau_sym": None,
            "tau_sev": None,
            "psi": None,
        }
        self.default_values = self._load_json(default_values_path)
        self.default_domains = self._load_json(default_domains_path)
        self.data_handler = DataHandler()

        self.update_params(params, fill_missing_values, reset=True)

        t, J, K = self._params["t"], self._params["J"], self._params["K"]

        self.PARAM_SHAPE = {
            "M": (J, K),
            "V": (J, K),
            "R": (J, K),
            "S": (J, K),
            "E": (J, K),
            "E_tr": (J, K),
            "E_nt": (J, K),
            "I": (J, K),
            "I_asym": (J, K),
            "I_sym": (J, K),
            "I_sev": (J, K),
            "Q": (J, K),
            "Q_asym": (J, K),
            "Q_sym": (J, K),
            "Q_sev": (J, K),
            "D": (J, K),
            "N": (J, K),
            "K": (0,),
            "J": (0,),
            "t": (0,),
            "basic_reprod_num": (t, J, K),
            "Beds": (1,),
            "beta_asym": (t, J, K, K),
            "beta_sym": (t, J, K, K),
            "beta_sev": (t, J, K, K),
            "gamma_asym": (t, J, K),
            "gamma_sym": (t, J, K),
            "gamma_sev": (t, J, K),
            "gamma_sev_r": (t, J, K),
            "gamma_sev_d": (t, J, K),
            "epsilon": (t, J, K),
            "mu_sym": (t, J, K),
            "mu_sev": (t, J, K),
            "nu": (t, J, K),
            "rho_mat": (t, J, K),
            "rho_vac": (t, J, K),
            "rho_rec": (t, J, K),
            "sigma": (t, J, K),
            "tau_asym": (t, J, K),
            "tau_sym": (t, J, K),
            "tau_sev": (t, J, K),
            "psi": (t, J, K),
        }

        self.key_list = [
            "M",
            "V",
            "R",
            "S",
            "E",
            "E_tr",
            "E_nt",
            "I",
            "I_asym",
            "I_sym",
            "I_sev",
            "Q",
            "Q_asym",
            "Q_sym",
            "Q_sev",
            "D",
            "N",
            "K",
            "J",
            "t",
            "basic_reprod_num",
            "Beds",
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

        # Set with update_shape_data()
        self.classes_keys = None
        self.classes_data = None  # [(J, K)]
        self.greeks_keys = None
        self.greeks_data = None  # [(t, J, K)]
        self.special_greeks_keys = None
        self.special_greeks_data = None  # [(t, J, K, K)]
        self.hyper_keys = None
        self.hyper_data = None  # [(0,)]
        self.misc_keys = None
        self.misc_data = None  # [(1,)]
        self.update_shape_data(t, J, K)

        # Data for direct use for the simulator (with usually solve_ivp)
        self.sim_data = None  # set in initialize_parameters() or manually with _create_sim_data()

    def update(self, params, fill_missing_values, reset=False) -> None:
        """
        Updates the controller to the latest parameters. 
        Wrapper around update_params and update_shape_data.
        """
        self.update_params(params, fill_missing_values, reset=reset)
        t, J, K = self._params["t"], self._params["J"], self._params["K"]
        # FIXME broken after init because Data is already in the long format
        # self.update_shape_data(t, J, K)

    def update_params(self, params, fill_missing_values, reset=False) -> None:
        """
        Updates the controller if new data/parameters is/are available. 
        Sets self._params to current parameter dict.
        """
        if reset:
            self.reset()

        if fill_missing_values:
            # build complete parameter set with controller
            # add default values to given params dict
            self.initialize_parameters(params)
        else:
            # add given params to already existing parameter dict
            self.check_params(params)
            self.set_params(params)

    def update_shape_data(self, t, J, K):
        shape_data_dict = self.broadcast_params_into_shape()
        # shapes taken from PARAM_SHAPE
        self.classes_data = shape_data_dict[(J, K)]
        self.greeks_data = shape_data_dict[(t, J, K)]
        self.special_greeks_data = shape_data_dict[(t, J, K, K)]
        self.hyper_data = shape_data_dict[(0,)]
        self.misc_data = shape_data_dict[(1,)]

    def broadcast_params_into_shape(self, params: dict = None, params_shapes: dict = None) -> dict:
        """
        Creates a 1D np.array for every unique shape in params_shapes. 
        E.g. all params with shape (J, K) are converted to an array with 
        shape (NumberOfClasses*J*K,). Can be reshaped back into an array with the first dimension
        being the parameter using np.reshape(X, NumberOfClasses, J, K).

        The order of the parameters in the output one dimensional array is based on the order
        they are in in params

        Parameters
        ----------
        params : dict, optional
            [description], by default None
        params_shapes : dict, optional
            [description], by default None

        Returns
        -------
        dict
            Dictionary with key being the shape and value being the
            one dimensional array of shape (NumberOfClasses*J*K,)
        """
        if not params:
            params = self._params
        if not params_shapes:
            params_shapes = self.PARAM_SHAPE

        # Parameters set to None
        none_keys = [
            key
            for key in params.keys()
            if not (isinstance(params[key], np.ndarray) or params[key])
        ]

        out_dict = {}
        unique_shapes = set(params_shapes.values())
        for shape in unique_shapes:
            # keys that have the selected shape and with their values not set to None
            keys_with_shape = [
                k for k in params if params_shapes[k] == shape and k not in none_keys
            ]

            # Save key lists to check order later if necessary
            t, J, K = params["t"], params["J"], params["K"]
            if shape == (J, K):
                self.classes_keys = keys_with_shape
            elif shape == (t, J, K):
                self.greeks_keys = keys_with_shape
            elif shape == (t, J, K, K):
                self.special_greeks_keys = keys_with_shape
            elif shape == (0,):
                self.hyper_keys = keys_with_shape
            elif shape == (1,):
                self.misc_keys = keys_with_shape

            # final shape of the output array, that can be
            # reshaped into (len(keys_with_shape), shape[0], shape[1], ...)
            final_1D_shape = (reduce(lambda x, y: x * y, [len(keys_with_shape)] + list(shape)),)
            shape_data = np.ones(final_1D_shape)

            # Input every broadcasted value into final output array shape_data
            for i in range(len(keys_with_shape)):
                left = reduce(lambda x, y: x * y, [i] + list(shape))
                right = reduce(lambda x, y: x * y, [i + 1] + list(shape))
                shape_data[left:right] = (np.ones(shape) * params[keys_with_shape[i]]).ravel()

            out_dict[shape] = shape_data

        return out_dict

    def reset(self):
        """
        Resets the current controllers paramter dict to None
        """
        for key in self._params:
            self._params[key] = None

    def _load_json(self, path) -> dict:
        try:
            with open(path) as f:
                file_ = json.load(f)
        except FileExistsError as fee:
            raise fee
        if isinstance(file_, dict):
            return file_
        else:
            raise FileNotFoundError(f"{path} does not contain any values.")

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
            if type(value) == np.ndarray:
                for val in value:
                    if domain[0] > val or val > domain[1]:
                        return False
                return True
            else:
                if domain[0] <= value <= domain[1]:
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
        for attr, val_list in params.items():
            if isinstance(val_list, list):
                for val in val_list:
                    if not self.valid_domain(attr, val):
                        raise AttributeError(
                            f"Invalid attribute, {attr} can not be of value {val}. "
                            "Check domain definition."
                        )

        # make sure params[J] (entities number) and params[K] (age group number) are available
        try:
            params["J"]
        except AttributeError as ae:
            # TODO logging info or raise Error
            self._params["J"] = self.default_values["J"]

        try:
            params["K"]
        except AttributeError as ae:
            # TODO logging info
            for key in params:
                if key not in ["K", "J", "beta"]:
                    self._params["K"] = len(params[key][0])
                    break

        # FIXME Dimensions see issue #28
        # make sure entities and age groups are correct in every parameter
        # for key in params:
        #     if key not in ["K", "J", "beta"]:
        #         if len(params[key][0]) != params["K"]:
        #             raise ValueError(
        #                 f"K:{self._K} does not match the first dimension of the"
        #                 " parameters. If you intended to not subdivide the different classes,"
        #                 " choose K=1 and wrap your parameters in a list."
        #             )
        #     elif key == "beta":
        #         # TODO check correct shape of beta
        #         pass

        #     # check if classes add up to one
        #     one = 1.0
        #     for key in {
        #         "M",
        #         "V",
        #         "S",
        #         "E",
        #         "E2",
        #         "I",
        #         "I2",
        #         "I3",
        #         "Q",
        #         "Q2",
        #         "Q3",
        #         "R",
        #         "D",
        #     } & set(params.keys()):
        #         one -= params[key]
        # TODO temporarily outcommented; check class domains
        # if not -1.0e-14 < one < 1.0e-14:
        #     raise ValueError(
        #         "Epidemiological classes do not add up to one." "Check input parameters."
        #     )

    def initialize_parameters(self, params: dict = None, load_base_data: bool = False) -> dict:
        """
        Initializes all parameters either by default or by setting it with supplied params dict
        """
        # Initialize all parameters with their start _0 value;
        if not params:
            params = {}
        for key, val in self.default_values.items():
            if key in params:
                self._params[key] = params[key]
            else:
                self._params[key] = val

        # TODO load data from DataHandler and put them into params
        if load_base_data:
            base_data = self.data_handler.get_simulation_initial_values()
            if len(base_data) > 0:
                self._params["N"], self._params["Beds"] = [], []
                map_params = {}
                for i, (key, value) in enumerate(base_data.items()):
                    self._params["N"].append(value["N"])
                    self._params["Beds"].append(value["B"])
                    map_params[f"{i}"] = key

        # make sure types are clear first under valid_domain and then initialize within bounds
        self.check_params(self._params)

        return self._params

    def set_params(self, params: dict) -> None:
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
        for key in params:
            self._params[key] = params[key]
        self.check_params(self._params)
        self.sim_data = self._create_sim_data()

    def get_params(self, keys: list = None) -> dict:
        """
        Prepares class attributes for retrieval and makes sure values are valid

        Parameters
        ----------
            keys : list
                list of strings, that are keys for parameters
        """
        if keys:
            return {key: self._params[key] for key in keys}
        else:
            return self._params
