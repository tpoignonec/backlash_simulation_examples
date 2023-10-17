# -*- coding: utf-8 -*-

# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

import numpy as np
#from copy import deepcopy
from .generic_backlash import AbstractBacklash
#from typing import Callable


class PassThrough (AbstractBacklash):
    """ Dummy object that is just a pass through element """
    def __init__(self, 
                 q_label : str = 'undefined', q_unit : str = 'undefined', 
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        super().__init__( parameters_count = 0,
                 parameters_labels = [], parameters_units = [],
                 q_label = q_label, q_unit = q_unit, 
                 c_label = c_label, c_unit = c_unit 
        )
        self._f = lambda x_k_1_, q_k_: x_k_1_
        self._F = np.array([[1]])
        self._map_x_to_c = np.array([[1]])
        self._obs = np.array([[1]])        
        self._active_branch = "pass-through"
        
        self._model_name = "Pass-through (dummy) component"        
        
    def evaluate(self, x_k_1 : np.ndarray, q_k, q_k_1 = None) -> bool:
        # Nothing to to
        return True
    
    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        return 0.0
        
#--------------------------------------------------------------------------------------------------
class BacklashConstWidthOneDof (AbstractBacklash):
    """ 
    1 DOF backlash model whose parameter is the (constant) backlash width. The slope of the characteristic is fixed and equal to 1.
    """
    def __init__(self, 
                 q_label : str = 'undefined', q_unit : str = 'undefined', 
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        # Invoke parent constructor (generic task model)
        super().__init__(parameters_count = 1, 
                         parameters_labels = ["bachlash witdh"], 
                         parameters_units =  [q_unit],
                         q_label = q_label, q_unit = q_unit, 
                         c_label= c_label, c_unit = c_unit
        )
        
        # Set static properties                 
        self._map_x_to_c = np.array([1,0]).reshape((1,-1)) # c = H_c @ X
        
        # Update module name
        self._model_name = "1 dof constant width backlash (with slope = 1)"
                
    def evaluate(self, x_k_1 : np.ndarray, q_k : np.ndarray, q_k_1 : np.ndarray = None) -> np.ndarray:
        x_k_1_ = x_k_1.reshape((-1))
        q_k_ = float(q_k)
        q_k_1_ = float(q_k_1)

        f_b = None
        F_b = None
        
        # Left smooth branch (L)
        if (q_k_ <= x_k_1_[0] -  x_k_1_[1]/2) and ( (q_k_<= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "left"
            f_b = lambda x_k_1_, q_k_: q_k_+ x_k_1_[1]/2
            F_b = np.array([[0, 1/2]])
            self._obs = np.array([1, 1])
        # Right smooth branch (R)                
        elif (q_k_>= x_k_1_[0] +  x_k_1_[1]/2) and ( (q_k_>= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "right"
            f_b = lambda x_k_1_, q_k_: q_k_- x_k_1_[1]/2 
            F_b = np.array([[0, -1/2]])
            self._obs = np.array([1,1])
        # Deadzone branch (DZ)          
        else :
            self._active_branch = "deadzone"
            f_b = lambda x_k_1_, q_k_:  x_k_1_[0]
            F_b = np.array([[1, 0]])
            self._obs = np.array([1,0])   
        # Store results and return
        self._f = lambda x_k_1_, q_k_: np.append(f_b(x_k_1_, q_k_), x_k_1_[1:])
        self._F = np.append(F_b, np.array([[0,1]]), axis=0)        
        return self._f(x_k_1_, q_k_)

    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        return float(x_k[1])
    
    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        return 1.0
        
    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        return 0.0
           
            
#--------------------------------------------------------------------------------------------------
class BacklashConstWidthAndSlope (AbstractBacklash):
    """ 
    2 DOF backlash model whose parameters are the (constant) backlash width and the slope of the characteristic.
    """
    def __init__(self, 
                 q_label : str = 'undefined', q_unit : str = 'undefined', 
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        # Invoke parent constructor (generic task model)
        super().__init__(parameters_count = 2, 
                         parameters_labels = ["bachlash witdh", "slope"], 
                         parameters_units =  [q_unit, c_unit+"/"+q_unit],
                         q_label = q_label, q_unit = q_unit, 
                         c_label= c_label, c_unit = c_unit
        )
        
        # Set static properties                 
        self._map_x_to_c = np.array([1,0,0]).reshape((1,-1)) # c = H_c @ X
        
        # Update module name
        self._model_name = "2 dof constant width backlash and tunable slope"
                
    def evaluate(self, x_k_1 : np.ndarray, q_k : np.ndarray, q_k_1 : np.ndarray = None) -> np.ndarray:
        x_k_1_ = x_k_1.reshape((-1))
        q_k_ = float(q_k)
        q_k_1_ = float(q_k_1)

        f_b = None
        F_b = None
        
        # Left smooth branch (L)
        if (q_k_ <= x_k_1_[0]/x_k_1_[2] -  x_k_1_[1]/2) and ( (q_k_<= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "left"
            f_b = lambda x_k_1_, q_k_: (q_k_+ x_k_1_[1]/2)*x_k_1_[2]
            F_b = np.array([[0, x_k_1_[2]/2, q_k_+ x_k_1_[1]/2]])
            self._obs = np.array([1, 1,1])
        # Right smooth branch (R)                
        elif (q_k_>= x_k_1_[0]/x_k_1_[2]  +  x_k_1_[1]/2) and ( (q_k_>= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "right"
            f_b = lambda x_k_1_, q_k_: (q_k_- x_k_1_[1]/2)*x_k_1_[2] 
            F_b = np.array([[0,-x_k_1_[2]/2, q_k_-x_k_1_[1]/2]])
            self._obs = np.array([1,1,1])
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
        return float(x_k[1])
    
    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        return float(x_k[2])
        
    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        return 0.0
    
    


#--------------------------------------------------------------------------------------------------
class BacklashConstWidthAndSlopeAndOffset (AbstractBacklash):
    """ 
    3 DOF backlash model whose parameters are the (constant) backlash width and the slope of the characteristic.
    """
    def __init__(self, 
                 q_label : str = 'undefined', q_unit : str = 'undefined', 
                 c_label :str = 'undefined', c_unit : str = 'undefined') :
        # Invoke parent constructor (generic task model)
        super().__init__(parameters_count = 3, 
                         parameters_labels = ["C_L", "C_R", "slope"], 
                         parameters_units =  [q_unit, q_unit, c_unit+"/"+q_unit],
                         q_label = q_label, q_unit = q_unit, 
                         c_label= c_label, c_unit = c_unit
        )
        
        # Set static properties                 
        self._map_x_to_c = np.array([1,0,0,0]).reshape((1,-1)) # c = H_c @ X
        
        # Update module name
        self._model_name = "3 dof constant width backlash with tunable slope and q offset"
                
    def evaluate(self, x_k_1 : np.ndarray, q_k : np.ndarray, q_k_1 : np.ndarray = None) -> np.ndarray:
        x_k_1_ = x_k_1.reshape((-1))
        q_k_ = float(q_k)
        q_k_1_ = float(q_k_1)

        f_b = None
        F_b = None
        
        # Left smooth branch (L)
        if (q_k_ <= x_k_1_[0]/x_k_1_[3] +  x_k_1_[1]) and ( (q_k_<= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "left"
            f_b = lambda x_k_1, q_k : (q_k - x_k_1[1])*x_k_1[3]
            F_b = np.array([[0, -x_k_1[3], 0, q_k-x_k_1[1]]])
            self._obs = np.array([1, 1,0,1])
        # Right smooth branch (R)                
        elif (q_k_>= x_k_1_[0]/x_k_1_[3]  +  x_k_1_[2]) and ( (q_k_>= q_k_1_) or (q_k_1_ is None)) :
            self._active_branch = "right"
            f_b = lambda x_k_1, q_k : (q_k - x_k_1[2])*x_k_1[3] 
            F_b =  np.array([[0,0,-x_k_1[3], q_k-x_k_1[2]]])
            self._obs =  np.array([1,0,1,1])
        # Deadzone branch (DZ)          
        else :
            self._active_branch = "deadzone"
            f_b = lambda x_k_1, q_k :  x_k_1[0]
            F_b = np.array([[1, 0, 0, 0]])
            self._obs =np.array([1,0,0,0])
        # Store results and return
        self._f = lambda x_k_1, q_k : np.append(f_b(x_k_1, q_k), x_k_1[1:])
        self._F = np.append(F_b, np.array([[0,1,0,0],[0,0,1,0], [0,0,0,1]]), axis=0)    
        return self._f(x_k_1_, q_k_)

    def equivalent_backlash_width(self, x_k : np.ndarray) -> float:
        return float(np.abs(x_k[1]-x_k[2]))    
    
    def equivalent_mean_slope(self, x_k : np.ndarray) -> float:
        return float(np.abs(x_k[3]))
    
    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> float:
        return float((x_k[1]+x_k[2])/2)    


#----------------------------------------------------------------------------------------

def get_backlash_template(constanst_width : bool, tunable_slope : bool, tunable_offset : bool, verbose = True) :
    chosen_model = None
    if constanst_width and not tunable_slope and not tunable_offset :        
        chosen_model = BacklashConstWidthOneDof
    elif constanst_width and tunable_slope and not tunable_offset :
        chosen_model = BacklashConstWidthAndSlope
    elif constanst_width and tunable_slope and tunable_offset :
        chosen_model = BacklashConstWidthAndSlopeAndOffset
    elif (not constanst_width) and tunable_slope and tunable_offset :
        raise Exception('4 dof model: TODO!')
    else :
        raise Exception('Invalid request')

    if verbose :
        print("Model's name: ", chosen_model.__name__)
        print("Model's description: ", chosen_model.__doc__)

    return chosen_model