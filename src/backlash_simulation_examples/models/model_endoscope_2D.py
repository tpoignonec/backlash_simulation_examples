# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

"""
Created on Thu Sep  8 15:31:46 2022

@author: Thiba
"""
import numpy as np
import matplotlib.pyplot as plt

# Symbolic utils
from sympy import *

epsilon_B = 1e-5

def safe_B(B) :
    if np.abs(B) < epsilon_B :
        if np.abs(B) == 0 :
            sign_value = 1
        else :
            sign_value = np.sign(B)
        return epsilon_B * sign_value
    else :
        return B

def build_functions(d = 0.02, L = 0.05):
    sym_c = MatrixSymbol('c', 2, 1)
    sym_f_kinematics = Matrix([ [L/sym_c[1,0] * (1-cos(sym_c[1,0])) + d * sin(sym_c[1,0])],
                                [sym_c[0,0] + L/sym_c[1,0] * sin(sym_c[1,0]) + d * cos(sym_c[1,0])]])
    sym_f_jacobian = sym_f_kinematics.jacobian(sym_c)

    f_kinematics_ = lambdify(sym_c, sym_f_kinematics, "numpy")
    f_jacobian_ = lambdify(sym_c, sym_f_jacobian, "numpy")

    def f_kinematics(c_) :
        c = c_.reshape((-1,1)).astype(float)
        t = float(c.reshape(-1)[0]) # linear position
        B = float(c[1]) # Bending angle
        c[1,0] = safe_B(B)
        return f_kinematics_(c)

    def f_jacobian(c_) :
        c = c_.reshape((-1,1)).astype(float)
        t = float(c.reshape(-1)[0]) # linear position
        B = float(c[1]) # Bending angle
        c[1,0] = safe_B(B)
        return f_jacobian_(c)

    return f_kinematics, f_jacobian

f_kinematics, f_jacobian = build_functions()

test_singularity = f_kinematics(np.array([0.,0.]))

def draw_robot(c, d = 0.02, L = 0.05) :
    t = float(c.reshape(-1)[0]) # linear position
    B = float(c[1]) # Bending angle
    B = safe_B(B)
    pts = []
    # draw base
    for alpha in np.linspace(0,1,20) :
        pts.append(np.array([[0],[alpha*t]]))
    # draw bending section -> TODO
    # Draw tip
    for alpha in np.linspace(0,1,20) :
        dummy_d = alpha*d
        point = np.array([ [L/B * (1-cos(B)) + dummy_d * sin(B)],
                                    [t + L/B * sin(B) + dummy_d * cos(B)]])
        pts.append(point)
    return  np.asarray(pts).reshape((-1,2))


if __name__ == "__main__":
    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.grid()
    pts_p = []
    t= 0.07
    for B in np.linspace(-np.pi/4,np.pi,6) :
        c = np.array([[t],[B]])
        pts_p.append(f_kinematics(c))

        robot_pts = draw_robot(c)
        ax.plot(robot_pts[:,0], robot_pts[:,1])

    pts_p = np.asarray(pts_p).reshape((-1,2))
    ax.scatter(pts_p[:,0], pts_p[:,1])


