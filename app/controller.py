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
    self._I_asym0_rel = None
    self._I_sym = None
    self._I_sym0_rel = None
    self._I_sev = None
    self._I_sev0_rel = None

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
        load_base_parameter: bool = True,
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
            "I_asym0_rel": None,
            "I_sym": None,
            "I_sym0_rel": None,
            "I_sev": None,
            "I_sev0_rel": None,
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
        self.map_params = {}
        self.data_handler = DataHandler()

        self.update_params(params, fill_missing_values, reset=True)

        if load_base_parameter:
            self._load_base_parameter()

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
            "I_asym0_rel": (J, K),
            "I_sym": (J, K),
            "I_sym0_rel": (J, K),
            "I_sev": (J, K),
            "I_sev0_rel": (J, K),
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
            "Beds": (J,),
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
            "I_asym0_rel",
            "I_sym",
            "I_sym0_rel",
            "I_sev",
            "I_sev0_rel",
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
        self.original_classes_keys = None
        self.original_classes_data = None  # classes_data before the first run (J, K)
        self.greeks_keys = None
        self.greeks_data = None  # [(t, J, K)]
        self.special_greeks_keys = None
        self.special_greeks_data = None  # [(t, J, K, K)]
        self.hyper_keys = None
        self.hyper_data = None  # [(0,)]
        self.misc_keys = None
        self.misc_data = None  # [(1,)]
        self.update_shape_data(self._params, t, J, K, init=True)

    def update(self, params, fill_missing_values, reset=False) -> None:
        """
        Updates the controller to the latest parameters. 
        Wrapper around update_params and update_shape_data.
        """
        # Last values
        last_params = {k: v[-1] for k, v in params.items()}
        self.update_params(last_params, fill_missing_values, reset=reset)

        # Full values over time
        t, J, K = self._params["t"], self._params["J"], self._params["K"]
        self.update_shape_data(params=params, t=t, J=J, K=K)

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
            self.set_params(params)

    def update_shape_data(self, params, t, J, K, init=False):

        # Only for the first run from __init__
        if init:
            shape_data_dict = self.broadcast_params_into_shape()
            # shapes taken from PARAM_SHAPE
            self.classes_data = shape_data_dict[(J, K)]
            self.original_classes_keys = self.classes_keys.copy()
            self.original_classes_data = shape_data_dict[(J, K)]
            self.greeks_data = shape_data_dict[(t, J, K)]
            self.special_greeks_data = shape_data_dict[(t, J, K, K)]
            self.hyper_data = shape_data_dict[(0,)]
            self.misc_data = shape_data_dict[(J,)]
        else:
            # new classes_data
            self.classes_keys = np.array([k for k in params.keys()])
            params_shapes = dict(zip(self.classes_keys, [(t, J, K)] * len(self.classes_keys)))
            self.classes_data = self.broadcast_params_into_shape(
                params=params, params_shapes=params_shapes, t=t, J=J, K=K,
            )[(t, J, K)]

            # check correct form/ parameter list for original_classes_data
            original_classes_data_dict = dict(
                zip(
                    self.original_classes_keys,
                    self.original_classes_data.reshape((len(self.original_classes_keys), J, K)),
                )
            )
            new_original_classes_data = {}
            for i, key in enumerate(self.classes_keys):
                if key in original_classes_data_dict:
                    new_original_classes_data[key] = original_classes_data_dict[key]
                else:
                    new_original_classes_data[key] = self.classes_data.reshape(
                        (self.classes_keys, J, K)
                    )[i]

            self.original_classes_keys = new_original_classes_data.keys()
            self.original_classes_data = np.array(list(new_original_classes_data.values())).ravel()

    def broadcast_params_into_shape(
        self, params: dict = None, params_shapes: dict = None, t=None, J=None, K=None
    ) -> dict:
        """
        Creates a 1D np.array for every unique shape in params_shapes. 
        E.g. all params with shape (J, K) are converted to an array with 
        shape (NumberOfClasses*J*K,). Can be reshaped back into an array with the first dimension
        being the parameter using np.reshape(X, (NumberOfClasses, J, K)).

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
            if t == None or J == None or K == None:
                t, J, K = params["t"], params["J"], params["K"]
            if shape == (J, K):
                self.classes_keys = keys_with_shape
            elif shape == (t, J, K):
                # HACK
                if "S" not in keys_with_shape:
                    self.greeks_keys = keys_with_shape
            elif shape == (t, J, K, K):
                self.special_greeks_keys = keys_with_shape
            elif shape == (0,):
                self.hyper_keys = keys_with_shape
            elif shape == (J,):
                self.misc_keys = keys_with_shape

            # final shape of the output array, that can be
            # reshaped into (len(keys_with_shape), shape[0], shape[1], ...)
            final_1D_shape = (reduce(lambda x, y: x * y, [len(keys_with_shape)] + list(shape)),)
            shape_data = np.ones(final_1D_shape)

            # Input every broadcasted value into final output array shape_data
            for i in range(len(keys_with_shape)):
                left = reduce(lambda x, y: x * y, [i] + list(shape))
                right = reduce(lambda x, y: x * y, [i + 1] + list(shape))
                # Already in correct shape
                param = np.array(params[keys_with_shape[i]])
                if param.shape == shape:
                    shape_data[left:right] = param.ravel()
                else:
                    shape_data[left:right] = (np.ones(shape) * param).ravel()

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
                for val in value.ravel():
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

        self._update_i()
        # make sure types are clear first under valid_domain and then initialize within bounds
        self.check_params(self._params)

        return self._params

    def _load_base_parameter(self):
        base_data = self.data_handler.get_simulation_initial_values(
            number_of_districts=self._params["J"]
        )
        if len(base_data) > 0:
            temp_params = {"N": [], "Beds": []}
            for i, (key, value) in enumerate(base_data.items()):
                temp_params["N"].append(value["N"])
                temp_params["Beds"].append(value["B"])
                if 10000 > key > 0:
                    self.map_params[i] = f"0{key}"
                else:
                    self.map_params[i] = f"{key}"

        self._params["N"] = np.array(temp_params["N"], dtype=np.float64)
        self._params["Beds"] = np.array(temp_params["Beds"], dtype=np.float64)
        self._update_i()
        self._params["S"] = (
            self._params["N"]
            - self._params["I_asym"]
            - self._params["I_sym"]
            - self._params["I_sev"]
        )

    def _update_i(self):
        if len(self._params["I_asym0_rel"]) > 0 and self._params["I_asym0_rel"][0] != 0:
            self._params["I_asym"] = self._params["N"] * np.array(self._params["I_asym0_rel"])
        if len(self._params["I_sym0_rel"]) > 0 and self._params["I_sym0_rel"][0] != 0:
            self._params["I_sym"] = self._params["N"] * np.array(self._params["I_sym0_rel"])
        if len(self._params["I_sev0_rel"]) > 0 and self._params["I_sev0_rel"][0] != 0:
            self._params["I_sev"] = self._params["N"] * np.array(self._params["I_sev0_rel"])

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
            # Consider shape of parameter and set new values to uniformally distributed value
            # TODO find a more accurate extrapolation method to conserve relations in the param
            target_len = len(self._params[key])
            self._params[key] = [params[key] for _ in range(target_len)]

            # HACK Better way to update shape data for single parameter?
            t, J, K = self._params["t"], self._params["J"], self._params["K"]
            # Overwriting a single parameter in shapes data
            if key in self.greeks_keys:
                ix = self.greeks_keys.index(key)
                greeks_data = self.greeks_data.reshape((len(self.greeks_keys), t, J, K))
                greeks_data[ix] = np.ones((t, J, K)) * params[key]
                self.greeks_data = greeks_data.ravel()

            elif key in self.special_greeks_keys:
                ix = self.special_greeks_keys.index(key)
                special_greeks_data = self.special_greeks_data.reshape(
                    (len(self.special_greeks_keys), t, J, K, K)
                )
                special_greeks_data[ix] = np.ones((t, J, K, K)) * params[key]
                self.special_greeks_data = special_greeks_data.ravel()

        self.check_params(self._params)

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

    def get_full_params(self, use_original_classes_data: bool = True) -> dict:
        """
        Reshapes and combines classes_data, greeks_data, hyper_data and misc_data.

        Returns
        -------
        dict
            Full paramter with full shape e.g. (t, J, K)
        """

        t, J, K = self._params["t"], self._params["J"], self._params["K"]

        if use_original_classes_data:
            classes_data = self.original_classes_data.reshape(
                (len(self.original_classes_keys), J, K)
            )
        else:
            classes_data = self.classes_data.reshape((len(self.classes_keys), J, K))

        greeks_data = self.greeks_data.reshape((len(self.greeks_keys), t, J, K))
        special_greeks_data = self.special_greeks_data.reshape(
            (len(self.special_greeks_keys), t, J, K, K)
        )
        misc_data = self.misc_data.reshape((len(self.misc_keys), J,))

        params = dict(
            zip(
                (
                    *self.classes_keys,
                    *self.greeks_keys,
                    *self.special_greeks_keys,
                    *self.misc_keys,
                ),
                (*classes_data, *greeks_data, *special_greeks_data, *misc_data),
            )
        )
        params["J"] = J
        params["K"] = K
        params["t"] = t
        params["N"] = np.ones((J, K)) * self._params["N"]

        return params
