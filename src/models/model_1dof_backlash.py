# -*- coding: utf-8 -*-

# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

"""
Created on Thu Sep 15 03:26:33 2022

@author: Thiba
"""

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (13,6)

def eval_1DOF_width_and_slope (x_k_1_, q_k, q_k_1 = None) :
    '''    
    Two parameters backlash model of constant width.
    
    Parameters
    ----------
    x_k_1_ : np.array
        previous state. 
        Content: x = [c, b, m]^T where b is the backlash witdh and m is the slope of the backlash caracteristic.
    q_k : float
        current actuator position.
    q_k_1 : float, optional
        Previous  actuator position. The default is None (not used in that case!!!).

    Returns
    -------
    f : function
        state transition function x_k = f(x_k_1, q_k).
    F : np.array
        state transition matrix df/dx.
    H_c : np.array
        map from state to c variables: c = H_c @ x
    x_obserbable : np.array
        boolean array, 1 if variable is observable, o otherwise.
    '''
    x_k_1 = x_k_1_.reshape((-1))
    x_obserbable = None
    F_b = None
    # Left smooth branch (A)
    if (q_k <= x_k_1[0]/x_k_1[2] -  x_k_1[1]/2) and ( (q_k <= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k + x_k_1[1]/2)*x_k_1[2]
        F_b = np.array([[0, x_k_1[2]/2, q_k + x_k_1[1]/2]])
        x_obserbable = np.array([1, 1,1])
    # Right smooth branch (B)                
    elif (q_k >= x_k_1[0]/x_k_1[2]  +  x_k_1[1]/2) and ( (q_k >= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[1]/2)*x_k_1[2] 
        F_b = np.array([[0,-x_k_1[2]/2, q_k-x_k_1[1]/2]])
        x_obserbable = np.array([1,1,1])
     # Deadzone branch (C)          
    else :
        f_b = lambda x_k_1, q_k :  x_k_1[0]
        F_b = np.array([[1, 0, 0]])
        x_obserbable = np.array([1,0,0])   
    f = lambda x_k_1, q_k : np.append(f_b(x_k_1, q_k), x_k_1[1:])
    F = np.append(F_b, np.array([[0,1,0],[0,0,1]]), axis=0)
    H_c = np.array([1,0,0]).reshape((1,-1)) # c = H_c @ X
    return f, F, H_c, x_obserbable 

def eval_1DOF_backlash_slope_is_1 (x_k_1_, q_k, q_k_1 = None) :
    x_k_1 = x_k_1_.reshape((-1))
    x_obserbable = None
    F_b = None
    # Left smooth branch (A)
    if (q_k <= x_k_1[0] +  x_k_1[1]) and ( (q_k <= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : q_k - x_k_1[1]
        F_b = np.array([[0, -1, 0]])
        x_obserbable = np.array([1, 1,0])
    # Right smooth branch (B)        
    elif (q_k >= x_k_1[0]  +  x_k_1[2]) and ( (q_k >= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[2])
        F_b = np.array([[0,0,-1]])
        x_obserbable = np.array([1,0,1])
     # Deadzone branch (C)  
    else :
        f_b = lambda x_k_1, q_k :  x_k_1[0]
        F_b = np.array([[1, 0, 0]])
        x_obserbable = np.array([1,0,0])   
    f = lambda x_k_1, q_k : np.append(f_b(x_k_1, q_k), x_k_1[1:])
    F = np.append(F_b, np.array([[0,1,0],[0,0,1]]), axis=0)
    H_c = np.array([1,0,0]).reshape((1,-1)) # c = H_c @ X
    return f, F, H_c, x_obserbable 

def eval_1DOF_backlash_with_slope (x_k_1_, q_k, q_k_1 = None) :
    x_k_1 = x_k_1_.reshape((-1))
    x_obserbable = None
    F_b = None
    # Left smooth branch (A)
    if (q_k <= x_k_1[0]/x_k_1[3] +  x_k_1[1]) and ( (q_k <= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[1])*x_k_1[3]
        F_b = np.array([[0, -x_k_1[3], 0, q_k-x_k_1[1]]])
        x_obserbable = np.array([1, 1,0,1])
    # Right smooth branch (B)                
    elif (q_k >= x_k_1[0]/x_k_1[3]  +  x_k_1[2]) and ( (q_k >= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[2])*x_k_1[3] 
        F_b = np.array([[0,0,-x_k_1[3], q_k-x_k_1[2]]])
        x_obserbable = np.array([1,0,1,1])
     # Deadzone branch (C)          
    else :
        f_b = lambda x_k_1, q_k :  x_k_1[0]
        F_b = np.array([[1, 0, 0, 0]])
        x_obserbable = np.array([1,0,0,0])   
    f = lambda x_k_1, q_k : np.append(f_b(x_k_1, q_k), x_k_1[1:])
    F = np.append(F_b, np.array([[0,1,0,0],[0,0,1,0], [0,0,0,1]]), axis=0)
    H_c = np.array([1,0,0,0]).reshape((1,-1)) # c = H_c @ X
    return f, F, H_c, x_obserbable 

def eval_1DOF_backlash_with_2_slope (x_k_1_, q_k, q_k_1 = None) :
    x_k_1 = x_k_1_.reshape((-1))
    x_obserbable = None
    F_b = None
    # Left smooth branch (A)
    if (q_k <= x_k_1[0]/x_k_1[3] +  x_k_1[1]) and ( (q_k <= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[1])*x_k_1[3]
        F_b = np.array([[0, -float(x_k_1[3]), 0, q_k-float(x_k_1[1]), 0]])
        x_obserbable = np.array([1, 1,0,1,0])
    # Right smooth branch (B)                
    elif (q_k >= x_k_1[0]/x_k_1[4]  +  x_k_1[2]) and ( (q_k >= q_k_1) or (q_k_1 is None)) :
        f_b = lambda x_k_1, q_k : (q_k - x_k_1[2])*x_k_1[4] 
        F_b = np.array([[0, 0,-float(x_k_1[4]), 0, q_k-float(x_k_1[2])]])
        x_obserbable = np.array([1,0,1,0,1])
     # Deadzone branch (C)          
    else :
        f_b = lambda x_k_1, q_k :  x_k_1[0]
        F_b = np.array([[1, 0, 0, 0, 0]])
        x_obserbable = np.array([1,0,0,0,0])   
    f = lambda x_k_1, q_k : np.append(f_b(x_k_1, q_k), x_k_1[1:])
    F = np.append(F_b, np.array([[0,1,0,0, 0],[0,0,1,0,0], [0,0,0,1, 0], [0,0,0,0,1]]), axis=0)
    H_c = np.array([1,0,0,0,0]).reshape((1,-1)) # c = H_c @ X
    return f, F, H_c, x_obserbable 

def eval_1DOF_backlash(x_k_1_, q_k, q_k_1 = None, dof=3) :
    if (dof == 2) :
        return  eval_1DOF_backlash_slope_is_1(x_k_1_, q_k, q_k_1)
    elif (dof == 3) :
        return eval_1DOF_backlash_with_slope (x_k_1_, q_k, q_k_1)
    elif (dof == 4) :
        return eval_1DOF_backlash_with_2_slope (x_k_1_, q_k, q_k_1)
    else :
        raise RuntimeError('"dof" parameter does not match any template!')