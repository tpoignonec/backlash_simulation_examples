# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

import numpy as np
#from copy import deepcopy
from .generic_backlash import AbstractBacklash
#from typing import Callable

#--------------------------------------------------------------------------------------------------
class Backlash (AbstractBacklash):
    """
    2 + 2 DOF backlash model :
        - A fixed part composed of the "central fiber" :
            c = (q - C_O)m_O
        - An adaptive part encoding the actual backlash width
            b_width(c) = gamma * c + b_0
    Such that theta = [b_0, gamma]^T is the parameter vector
    --> X_k = [c_k, b_0k, gamma_k]^T
    """
    def __init__(self,
                 q_label : str = 'undefined', q_unit : str = 'undefined',
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        # Invoke parent constructor (generic task model)
        super().__init__(parameters_count = 2,
                         parameters_labels = ["bachlash width at c=0", "gamma"],
                         parameters_units =  [q_unit, q_unit+"/"+c_unit],
                         q_label = q_label, q_unit = q_unit,
                         c_label= c_label, c_unit = c_unit
        )

        # Set static properties
        self._map_x_to_c = np.array([1,0,0]).reshape((1,-1)) # c = H_c @ X

        # Fixed parameters (default values)
        self._C_0 = 0
        self._m_0 = 1.0

        # Update module name
        self._model_name = "2 + 2 dof backlash model. Tunablebacklash width (2 dof) + static affine TR"

    def set_fixed_parameters(self, m_0 = 1.0, C_0 = 0.0):
        self._C_0 = C_0
        self._m_0 = m_0

    def state_to_RL_representation(self, x_k) :
        return self.to_RL_representation(
            C_0 = self._C_0,
            m_0 = self._m_0,
            b_0 = x_k[1],
            gamma = x_k[2]
        )

    def to_RL_representation(self, C_0, m_0, b_0, gamma) :
        """
        Switch parameterization to the right/left branches model.

        Parameters
        ----------
        C_0 : float
            parameter value.
        m_0 : float
            parameter value.
        b_0 : float
            parameter value.
        gamma : float
            parameter value..

        Returns
        -------
        params : tuple
            (C_L, C_R, m_L, m_R)
        """
        C_L = C_0 - b_0/2.0
        C_R = C_0 + b_0/2.0
        m_L = m_0/(1.0 - m_0 * gamma/2)
        m_R = m_0/(1.0 + m_0 * gamma/2)
        return (C_L, C_R, m_L, m_R)

    def evaluate(self, x_k_1 : np.ndarray, q_k : np.ndarray, q_k_1 : np.ndarray = None) -> np.ndarray:
        x_k_1_ = x_k_1.reshape(-1)

        q_k_ = float(q_k)
        q_k_1_ = float(q_k_1)

        f_b = None
        F_b = None

        c_k_1_ = x_k_1_[0]
        C_L, C_R, m_L, m_R = self.state_to_RL_representation(x_k_1_)
        m_0 = self._m_0
        C_0 = self._C_0
        gamma = x_k_1_[2]

        # Left smooth branch (L)
        if (q_k_ <= c_k_1_/m_L + C_L) and ( (q_k_<= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "left"
            f_b = lambda x_k_1_, q_k_: \
                (q_k_ - (C_0 - x_k_1_[1]/2.0))*m_0/(1.0 - m_0 * x_k_1_[2]/2)
            F_b = np.array([[
                0,
                m_L/2,
                (q_k_ - C_L)*m_0/(2 * np.power(1-m_0*gamma/2, 2))
            ]])
            self._obs = np.array([1, 1,1])
        # Right smooth branch (R)
        elif (q_k_ >= c_k_1_/m_R + C_R) and ( (q_k_>= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "right"
            f_b = lambda x_k_1_, q_k_:  \
                (q_k_ - (C_0 + x_k_1_[1]/2.0))*m_0/(1.0 + m_0 * x_k_1_[2]/2)
            F_b = np.array([[
                0,
                -m_R/2,
                -(q_k_ - C_R)*m_0/(2 * np.power(1+m_0*gamma/2, 2))
            ]])
            self._obs = np.array([1, 1,1])
        # Deadzone branch (DZ)
        else :
            self._active_branch = "deadzone"
            f_b = lambda x_k_1_, q_k_:  x_k_1_[0]
            F_b = np.array([[1, 0, 0]])
            self._obs = np.array([1,0,0])
        # Store results and return
        self._f = lambda x_k_1_, q_k_: np.append(f_b(x_k_1_, q_k_), x_k_1_[1:])
        self._F = np.append(F_b, np.array([[0,1,0],[0,0,1]]), axis=0)
        return self._f(x_k_1_, q_k_)

    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        c = self._map_x_to_c @ x_k
        gamma = x_k[2]
        b_0 = x_k[1]
        return float(b_0 + c*gamma)

    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        return float(self._m_0)

    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        return float(self._C_0)

