# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

"""
Created on Thu Sep 15 02:38:12 2022

@author: Thibault Poignonec
"""

import numpy as np
import sys
# Symbolic utils
from sympy import *


def approx_jacobian(x,func,*args):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         x       - The state vector at which the Jacobian matrix is
desired list or np.array of shape ((-1,)) or ((-1,1))
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences

    """
    epsilon = np.sqrt(sys.float_info.epsilon)
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args)).reshape((-1,))
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(x0.shape)
    for i in range(len(x0)):
       dx[i] = epsilon
       jac[i] = (func(*((x0+dx,)+args)).reshape((-1,)) - func(*((x0-dx,)+args)).reshape((-1,)))/(2*epsilon)
       dx[i] = 0.0
    return jac.transpose()

class EndoscopicToolModel:
    def __init__(self, L = 0.05, d = 0.02) :
        # REF: Handbook of Robotic and Image-Guided Surgery. DOI: https://doi.org/10.1016/B978-0-12-814245-5.00008-6 (8.3.3)
        sym_c = MatrixSymbol('c', 3, 1)

        sym_f_kinematics = Matrix([
            [(L/sym_c[1,0] * (1-cos(sym_c[1,0])) + d * sin(sym_c[1,0]))*cos(sym_c[2,0])],
            [(L/sym_c[1,0] * (1-cos(sym_c[1,0])) + d * sin(sym_c[1,0]))*sin(sym_c[2,0])],
            [sym_c[0,0] + L/sym_c[1,0] * sin(sym_c[1,0]) + d * cos(sym_c[1,0])]])
        sym_f_jacobian = sym_f_kinematics.jacobian(sym_c)

        # For bending angle (c[1]) equal 0
        self.epsilon_singularity_B_ = np.pi/20

        sym_f_kinematics_straight_position = Matrix([
            [0],
            [0],
            [sym_c[0,0] + L + d]])

        sym_f_jacobian_straight_position = Matrix([
            [0., cos(sym_c[2,0])*(L/2+d),   0.],
            [0., sin(sym_c[2,0])*(L/2+d),   0.],
            [1., 0.,                        0.]])

        #Lambdify functions (using numpy)
        self.f_kinematics_ = lambdify(sym_c, sym_f_kinematics, "numpy")
        self.f_jacobian_ = lambdify(sym_c, sym_f_jacobian, "numpy")

        self.f_kinematics_straight_position_ = lambdify(sym_c, sym_f_kinematics_straight_position, "numpy")
        self.f_jacobian_straight_position_ = lambdify(sym_c, sym_f_jacobian_straight_position, "numpy")

    def f_kinematics(self, c_):
        # c = [translation, bending, rotation]
        c = c_.reshape((-1,1)).astype(float)
        B = float(c[1]) # Bending angle
        if B == 0 :
            return self.f_kinematics_straight_position_(c)
        else :
            return self.f_kinematics_(c)

    def f_jacobian(self, c_):
        # c = [translation, bending, rotation]
        c = c_.reshape((-1,1)).astype(float)
        B = float(c[1]) # Bending angle
        if B == 0 :
            return self.f_jacobian_straight_position_(c)
        else :
            return self.f_jacobian_(c)

    def observability_c(self, c_) :
        # c = [translation, bending, rotation]
        c = c_.reshape((-1,1)).astype(float)
        B = float(c[1]) # Bending angle
        if np.abs(B) < self.epsilon_singularity_B_ :
            return np.array([1,1,0]) # straight position -> rotation not observable
        elif np.abs(np.abs(B) - np.pi) < self.epsilon_singularity_B_ :
            return np.array([0,0,1]) # limit cartesian workspace B = +/- pi -> redundant translation with bending (confounded)
        else :
            return np.array([1,1,1])

if __name__ == "__main__" :
    model = EndoscopicToolModel()
    func_h = model.f_kinematics
    func_H = model.f_jacobian
    func_obs_c = model.observability_c

    print("h([1, 0, 0]) = \n", func_h(np.array([1,0,0])))
    print("h([1, pi/2, 0]) = \n", func_h(np.array([1,np.pi/2,0])))
    print("h([1, pi/2, pi]) = \n", func_h(np.array([1,np.pi/2,np.pi])))


    print("H([1, 0, 0]) = \n", func_H(np.array([1,0,0])))
    print("(numerical) H([1, 0, 0]) = \n", approx_jacobian(np.array([1,0,0]), model.f_kinematics))

    print("H([1, pi/2, 0]) = \n", func_H(np.array([1,np.pi/2,0])))
    print("(numerical) H([1, pi/2, 0]) = \n", approx_jacobian(np.array([1,np.pi/2,0]), model.f_kinematics))

    print("H([1, pi/2, pi/2]) = \n", func_H(np.array([1,np.pi/2,np.pi/2])))
    print("(numerical) H([1, pi/2, pi/2]) = \n", approx_jacobian(np.array([1,np.pi/2,np.pi/2]), model.f_kinematics))



    print("obs([1, 0, 0]) = \n", func_obs_c(np.array([1,0,0])))
    print("obs([1, pi/2, 0]) = \n", func_obs_c(np.array([1,np.pi/2,0])))
    print("obs([1, pi, 0]) = \n", func_obs_c(np.array([1,np.pi,0])))
    print("obs([1, -pi, 0]) = \n", func_obs_c(np.array([1,-np.pi,0])))
