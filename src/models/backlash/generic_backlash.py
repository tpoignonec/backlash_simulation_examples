# -*- coding: utf-8 -*-

# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

from abc import ABCMeta, abstractmethod
#from typing import Callable

import numpy as np
from copy import deepcopy
    
class AbstractBacklash(object):
    """
    A 1 DOF backlash abstract class that will be used for Backlash objects creation.
    """    
    __metaclass__ = ABCMeta    
    
    def __init__(self, 
                 parameters_count : int,
                 parameters_labels : list = None, parameters_units : list = None,
                 q_label : str = 'undefined', q_unit : str = 'undefined', 
                 c_label : str = 'undefined', c_unit : str = 'undefined' ) :
        
        self._params_count = parameters_count
        if parameters_labels is None :
            self._parameters_labels = ['undefined'] * parameters_count
        else :
            self._parameters_labels = deepcopy(parameters_labels)
              
        if parameters_labels is None :
            self._parameters_units = ['undefined'] * parameters_count
        else :
            self._parameters_units = deepcopy(parameters_units)
            
        
        if (len(self._parameters_labels) != len(self._parameters_units)) \
           or (len(self._parameters_labels) != self.parameters_count) :
               raise Exception("Invalid parameters passed to __init__().\
                            The labels and units list should be equal in lenght (and equal to parameters_count) !")
        
        self._q_label = deepcopy(q_label)
        self._q_unit = deepcopy(q_unit)
        self._q_range = (None, None)
        
        self._c_label = deepcopy(c_label)
        self._c_unit = deepcopy(c_unit)  
        self._c_range = (None, None)
        
        self._model_name = "unkwown"  
        
        self._active_branch = None
        self._f = None
        self._F = np.zeros((self.x_dim, self.x_dim)) * np.nan
        self._map_x_to_c = np.zeros((1, self.x_dim)) * np.nan
        self._obs = np.zeros((self.x_dim,)) * np.nan
    
    # ------------------
    # Static properties
    # ------------------    
    @property
    def parameters_labels(self) -> list:
        """ Short description of the parameters """
        return self._parameters_labels    
    
    @property
    def parameters_units(self) -> list:
        """ Units of the parameters """
        return self._parameters_units     
    
    @property
    def x_labels(self) -> list:
        """ Short description of the state X_i = [c_i, theta_i] where theta is the parameter vector."""
        return [self._c_label] + self._parameters_labels     
    
    @property
    def x_units(self) -> list:
        """ Units of the state X_i = [c_i, theta_i] where theta is the parameter vector."""
        return [self._c_unit] + self.parameters_units
    
    @property
    def x_dim(self) -> int:
        """ State variables vector dimension """
        return 1 + self.parameters_count    
    
    @property
    def parameters_count(self) -> int:
        """ Number of model parameters """
        return self._params_count     
    
    @property
    def model_name(self) -> str:
        """ Name of the model """
        return self._model_name     
    
    # ----------------------------
    # Dynamic/mutable properties
    # ----------------------------
    @property
    def q_range(self) -> tuple:         
        return self._q_range
    @q_range.setter
    def q_range(self, q_range) -> None:         
        self._q_range = deepcopy(q_range)
    
    @property
    def c_range(self) -> tuple:         
        return self._c_range
    @c_range.setter
    def c_range(self, c_range) -> None:         
        self._c_range = deepcopy(c_range)
        
    @property
    def active_branch(self) -> str:
        """ Name of the active branch. Raise error if not evaluated! """
        if self._active_branch is None :
            raise Exception("'active_branch' not initialized! Have you called 'evaluate()'?")            
        return self._active_branch
    
    @property
    def F(self) -> np.ndarray:
        '''    
        State transition matrix F = df/dx, where x = [c, theta].
        
        Returns
        -------
        F : np.ndarray
            dim(X) x dim(X) matrix
            
        Note
        ------
        Call 'evaluate()' first!!!
        '''
        if np.isnan(self._F).any() :
            raise Exception("F not initialized! Have you called 'evaluate()'?")
        return  self._F
    
    
    @property
    def map_x_to_c(self) -> np.ndarray:
        '''    
        State to c variable mapping such that c = map_x_to_c @ x
        
        Returns
        -------
        map_x_to_c : np.ndarray
            1 x dim(X) matrix
            
        Note
        ------
        Call 'evaluate()' first!!!
        '''
        if np.isnan(self._map_x_to_c).any() :
            raise Exception("'map_x_to_c' not initialized! Have you called 'evaluate()'?")
        return  self._map_x_to_c
    
    @property
    def obs(self) -> np.array:
        '''    
        Observability of state variables.
        
        Returns
        -------
        obs : np.array
            Vector of the same dimension as X (=1 + nb_param), obs[i] == True if the i_th state variable is observable.
            
        Note
        ------
        Call 'evaluate()' first!!!
        '''
        if np.isnan(self._obs).any() :
            raise Exception("'obs' not initialized! Have you called 'evaluate()'?")
        return  self._obs
    
    # -----------------------------
    # Methods and abstract methods
    # -----------------------------    
    def clear_dynamic_properties(self) :
        '''
        Set F, obs, and map_x_to_c to a matrix of NaNs
        Note
        -------
        Can be called in child class as 'super().clear_dynamic_properties()' to make sure obselete data is not kept.
        '''
        # Flag all properties as unvalid
        self._active_branch = None
        dim_x = 1 + self.params_count
        self._F = np.zeros((dim_x, dim_x)) * np.nan
        self._map_x_to_c = np.zeros((1, dim_x)) * np.nan
        self._obs = np.zeros((dim_x,)) * np.nan
        
    @abstractmethod
    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        ''' Evaluate the backlash width locally at c = c_k and return the value in whatever the units of q is.'''
        raise Exception("Not implemented!")     
        
    @abstractmethod
    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        ''' Evaluate the backlash's characteristic mean slope.'''
        raise Exception("Not implemented!")   
        
    @abstractmethod
    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        ''' Evaluate the backlash's characteristic mean q offset (i.e. mean value of q_k at c_k=0).'''
        raise Exception("Not implemented!")   
        
    @abstractmethod
    def evaluate(self, x_k_1 : np.ndarray, q_k : np.ndarray, q_k_1  : np.ndarray = None) -> np.ndarray:
        '''    
        Evaluate the state of the backlash element and returns True if successfull.
        This function should be call after each update of c OR theta!
        Once called, the active branch and associated utils/properties (f, F, Hc, etc.) can be retrieved.
        
        Parameters
        ----------
        x_k_1_ : np.array
            previous state. 
        q_k : float
            current actuator position.
        q_k_1 : float, optional
            Previous  actuator position. The default is None (not used in that case!!!).
        
        Returns
        -------
        x_k : np.ndarray
            current estimated state (or None if invalid model!)
        '''        
        return None
        
    
    @abstractmethod
    def f(self, x_k_1, q_k) -> np.ndarray :
        '''    
        State transition function.
        
        Returns
        -------
        f : Callable
            Function of the form x_k = f(x_k_1, q_k)
            
        Note
        ------
        Call 'evaluate()' first!!!
        '''    
        if self._f is None:
            raise Exception("'f' function undefined!")
        x_k_1_ = x_k_1.reshape((-1))
        q_k_ = float(q_k)
        return self._f(x_k_1_, q_k_)