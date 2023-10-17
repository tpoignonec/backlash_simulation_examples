# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

"""
Created on Tue Sep 20 10:49:17 2022

@author: Thiba
"""

import numpy as np
import sys
# Symbolic utils
from sympy import *
#import mygrad as mg # Automatic diff (see https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/AutoDiff.html)


epsilon_float_ = np.sqrt(sys.float_info.epsilon)
def approx_jacobian(x,func,epsilon = epsilon_float_,*args):
    """Approximate the Jacobian matrix of callable function func

       * Parameters
         x       - The state vector at which the Jacobian matrix is
desired
         func    - A vector-valued function of the form f(x,*args)
         epsilon - The peturbation used to determine the partial derivatives
         *args   - Additional arguments passed to func

       * Returns
         An array of dimensions (lenf, lenx) where lenf is the length
         of the outputs of func, and lenx is the number of

       * Notes
         The approximation is done using forward differences

    """
    x0 = np.asfarray(x)
    f0 = func(*((x0,)+args)).reshape((-1,))
    jac = np.zeros([len(x0),len(f0)])
    dx = np.zeros(x0.shape)
    for i in range(len(x0)):
       dx[i] = epsilon/2
       jac[i] = (func(*((x0+dx,)+args)).reshape((-1,)) - func(*((x0-dx,)+args)).reshape((-1,)))/(epsilon)
       dx[i] = 0.0
    return jac.transpose()

class EndoscopeTwoBendingModel:
    def __init__(self, L_f = 0.08, L_t = 0.02, D = 0.012) :
        self.L_f_ = L_f
        self.L_t_ = L_t
        self.D_ = D

        # Ref:
        #        - Ott et al. 2011 (TR-O)
        #        - Thesis manuscript of Rafael Porto, 2021 (see Chapter 2.3)

        # input -> c_cables = [delta_length_wire_1, delta_length_wire_2]
        # internal representation -> c_angles = [alpha, beta]

        # c_angles = [alpha, beta]
        self.sym_c_angles_ = MatrixSymbol('c_angles', 2, 1)
        sym_alpha = self.sym_c_angles_[0,0]
        sym_beta  = self.sym_c_angles_[1,0]

        # Forward kinematics
        self.sym_R_base_to_tip_ = Matrix([
            [Pow(sin(sym_alpha),2) + cos(sym_beta)*Pow(cos(sym_alpha),2),   -sin(sym_alpha)*cos(sym_alpha)*(1-cos(sym_beta)),               cos(sym_alpha)*sin(sym_beta)],
            [-sin(sym_alpha)*cos(sym_alpha)*(1-cos(sym_beta)),              Pow(cos(sym_alpha),2) + cos(sym_beta)*Pow(sin(sym_alpha),2),    sin(sym_alpha)*sin(sym_beta)],
            [-cos(sym_alpha)*sin(sym_beta),                                 -sin(sym_alpha)*sin(sym_beta),                                  cos(sym_beta)]
        ])
        self.sym_t_base_to_tip_ = Matrix([
            [(L_t * sin(sym_beta) + L_f/sym_beta * (1 - cos(sym_beta)) ) * cos(sym_alpha)],
            [(L_t * sin(sym_beta) + L_f/sym_beta * (1 - cos(sym_beta)) ) * sin(sym_alpha)],
            [L_t * cos(sym_beta) + L_f/sym_beta * sin(sym_beta)]
        ])
        self.sym_t_base_to_tip_beta_is_zero_ = Matrix([
            [0],
            [0],
            [L_t + L_f]
        ])
        self.func_R_base_to_tip_ = lambdify(self.sym_c_angles_, self.sym_R_base_to_tip_, "numpy")
        self.func_t_base_to_tip_ = lambdify(self.sym_c_angles_, self.sym_t_base_to_tip_, "numpy")
        self.func_t_base_to_tip_beta_is_zero_ = lambdify(self.sym_c_angles_, self.sym_t_base_to_tip_beta_is_zero_, "numpy")

        # Forward differential model
        self.sym_jacobian_t_base_to_tip_ = self.sym_t_base_to_tip_.jacobian(self.sym_c_angles_)
        self.sym_jacobian_t_base_to_tip_beta_is_zero_ = Matrix([
            [ 0,  cos(sym_alpha) * (L_f/2 + L_t)],
            [ 0,  sin(sym_alpha) * (L_f/2 + L_t)],
            [ 0.0,                          0.0]
        ])
        self.sym_jacobian_angular_vel_wrt_c_angle = Matrix([
            [-cos(sym_alpha) * sin(sym_beta),       -sin(sym_alpha)],
            [-sin(sym_alpha) * sin(sym_beta),       -cos(sym_alpha)],
            [ -1 + cos(sym_beta),                    0.0]
        ])
        self.func_jacobian_linear_vel_wrt_c_angles_ = lambdify(self.sym_c_angles_, self.sym_jacobian_t_base_to_tip_, "numpy")
        self.func_jacobian_linear_vel_wrt_c_angles_when_beta_is_zero_ = lambdify(self.sym_c_angles_, self.sym_jacobian_t_base_to_tip_beta_is_zero_, "numpy")
        self.func_jacobian_angular_vel_wrt_c_angle_ = lambdify(self.sym_c_angles_, self.sym_jacobian_angular_vel_wrt_c_angle, "numpy")

        # Derivatives matrix R
        sym_R_base_to_tip_derivative_wrt_alpha_ = diff(self.sym_R_base_to_tip_, sym_alpha)
        self.func_R_base_to_tip_derivative_wrt_alpha_ = lambdify(self.sym_c_angles_, sym_R_base_to_tip_derivative_wrt_alpha_, "numpy")

        sym_R_base_to_tip_derivative_wrt_beta_ = diff(self.sym_R_base_to_tip_, sym_beta)
        self.func_R_base_to_tip_derivative_wrt_beta_ = lambdify(self.sym_c_angles_, sym_R_base_to_tip_derivative_wrt_beta_, "numpy")

        self.camera_model_params_ = None
        self.init_default_camera_model()

    def init_default_camera_model(self):
        """ Default camera configuration (could not be meaningful, use for debug only...) """
        return self.init_camera_model (
            u_lims = (0, 720),
            v_lims = (0, 576),
            u0 = 349.6075,
            v0 = 285.0102,
            Kx = 423.9648,
            Ky = 454.2545,
            rotation_camera_wrt_bearing = 8.5*np.pi/180
        )

    def init_camera_model(self, u_lims, v_lims, u0, v0, Kx, Ky, rotation_camera_wrt_bearing = 0):
        """
        Init camera model parameters.
        The pin-hole model is of the form
            z = [z_x, z_y]^T = proj(P), in px,
        where
            z_x = (u0 + Kx*Px)/Pz
            z_y = (v0 + Ky*Py)/Pz

        Parameters
        ----------
        u_lims : tuple (float, float)
            limits image width in pixels.
        v_lims : tuple (float, float)
            limits image height in pixels.
        u0 : float
            Offset in image along u (i.e. center).
        v0 : float
            Offset in image along v (i.e. center).
        Kx : float
            Magnifying factor in x direction.
        Ky : float
            Magnifying factor in y direction.
        rotation_camera_wrt_bearing : float
            Opt. parameters to model incorrect camera orientation (Z orientation of F_c wrt tip frame)

        Returns
        -------
        bool
            OK

        """
        self.camera_model_params_ = {
            "u_lims" : u_lims,
            "v_lims" : v_lims,
            "u0" : u0,
            "v0" : v0,
            "Kx" : Kx,
            "Ky" : Ky,
            "rotation_camera_wrt_bearing" : rotation_camera_wrt_bearing,
            }
        return True

    # Kinematics
    def func_c_cables_to_c_angles_(self, c_cables_):
        c_cables = c_cables_.reshape((-1,)).astype(float)
        c_angles = np.zeros((2,1))
        D = self.D_
        if c_cables.any() :
            c_angles[0,0] = np.arctan2(-c_cables[1], -c_cables[0])
        c_angles[1,0] = (2/D)*np.sqrt(c_cables[0]**2 + c_cables[1]**2)
        return c_angles

    def func_jacobian_c_cables_to_c_angles_(self, c_cables_) :
        c_cables = c_cables_.reshape((-1,)).astype(float)
        D = self.D_
        return np.array([
            [-c_cables[1]/(c_cables[0]**2+c_cables[1]**2),              c_cables[0]/(c_cables[0]**2+c_cables[1]**2)],
            [(2/D)*c_cables[0]/np.sqrt(c_cables[0]**2+c_cables[1]**2),  (2/D)*c_cables[1]/np.sqrt(c_cables[0]**2+c_cables[1]**2)]
        ])

    def R_base_to_tip(self, c_cables_):
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        c_angles = self.func_c_cables_to_c_angles_(c_cables)

        angle_Z = self.camera_model_params_["rotation_camera_wrt_bearing"]
        mat_rotation_camera_wrt_bearing = np.array([
            [np.cos(angle_Z), -np.sin(angle_Z), 0.0],
            [np.sin(angle_Z),  np.cos(angle_Z), 0.0],
            [0.0,              0.0,             1.0]
        ])
        return self.func_R_base_to_tip_(c_angles) @ mat_rotation_camera_wrt_bearing

    def t_base_to_tip(self, c_cables_):
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        c_angles = self.func_c_cables_to_c_angles_(c_cables)
        if c_angles[1,0] == 0 :
            return self.func_t_base_to_tip_beta_is_zero_(c_angles)
        else :
            return self.func_t_base_to_tip_(c_angles)

    def jacobian_position (self, c_cables_, use_approx=False) :
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        if use_approx or (not c_cables.any()) :
            # c_cables = [0,0] -> model singularity (representation only)
            return approx_jacobian(c_cables, self.t_base_to_tip, epsilon=epsilon_float_)
        else :
            J_cable_to_angle = self.func_jacobian_c_cables_to_c_angles_(c_cables)
            c_angles = self.func_c_cables_to_c_angles_(c_cables)
            if c_angles[1,0] == 0 :
                J_angle_to_position = self.func_jacobian_linear_vel_wrt_c_angles_when_beta_is_zero_(c_angles)
            else :
                J_angle_to_position = self.func_jacobian_linear_vel_wrt_c_angles_(c_angles)
            return J_angle_to_position @ J_cable_to_angle

    def jacobian_angular (self, c_cables_) :
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        J_cable_to_angle = self.func_jacobian_c_cables_to_c_angles_(c_cables)
        c_angles = self.func_c_cables_to_c_angles_(c_cables)
        J_angle_to_position = self.func_jacobian_angular_vel_wrt_c_angle_ (c_angles)
        return J_angle_to_position @ J_cable_to_angle

    def full_jacobian(self, c_cables_) :
        J_position =  self.jacobian_position(c_cables_)
        J_angular =  self.jacobian_angular(c_cables_)
        J_cables_to_velocity_screw = np.concatenate([J_position, J_angular], axis=0)
        return J_cables_to_velocity_screw

    # Camera model
    def image_measurment_from_Fb(self, c_cables_, P_in_Fb) :
        b_R_c = self.R_base_to_tip(c_cables_)
        b_t_c = self.t_base_to_tip(c_cables_)
        P_in_Fc = b_R_c.T @ ( P_in_Fb.reshape((-1,1)) - b_t_c)
        return self.image_measurment_from_Fc(P_in_Fc)

    def image_measurment_from_Fc(self, P_in_Fc) :
        params = self.camera_model_params_
        p = P_in_Fc.reshape((-1,))
        Kx = params["Kx"]
        Ky = params["Ky"]
        return np.array([
            [params["u0"] + (Kx*p[0])/p[2]],
            [params["v0"] + (Ky*p[1])/p[2]]])

    def jacobian_image_measurement_wrt_c_and_P(self,  c_cables_, P_in_Fb_, approx=True) :
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        c_angles = self.func_c_cables_to_c_angles_(c_cables)
        # Camera model parameters
        camera_params = self.camera_model_params_
        Kx = camera_params["Kx"]
        Ky = camera_params["Ky"]

        # Pre compute robot tip orientation/position
        tip_orientation = self.R_base_to_tip(c_cables_)
        tip_position = self.t_base_to_tip(c_cables_)
        P_in_Fb = P_in_Fb_.reshape((-1,1))
        P_in_Fc = tip_orientation.T @ ( P_in_Fb - tip_position)

        # 1) jacobian wrt c
        # -K*(dp/dx * pz - px * dp/dz)/(dz^2)
        jacobian_image_measurement_wrt_c = np.zeros((2,2))
        jacobian_image_measurement_wrt_p = np.zeros((2,3))
        if approx or (not c_cables_.any()) :
            # c_cables = [0,0] -> model singularity (representation only)
            jacobian_image_measurement_wrt_c = approx_jacobian(c_cables, self.image_measurment_from_Fb, epsilon_float_, P_in_Fb)
            def image_measurment_from_Fb_P_as_first_arg(p, c) :
                return self.image_measurment_from_Fb(c, p)
            jacobian_image_measurement_wrt_p = approx_jacobian(P_in_Fb, image_measurment_from_Fb_P_as_first_arg, epsilon_float_, c_cables)
        else :
            print("WARNING!!! Not workingm to be fixed...")
            # Compute jacobian of P_in_F_c wrt alpha and beta
            J_angle_to_position = None
            if not c_cables_.any():
                J_angle_to_position = self.func_jacobian_linear_vel_wrt_c_angles_when_beta_is_zero_(c_cables)
            else :
                J_angle_to_position = self.func_jacobian_linear_vel_wrt_c_angles_(c_angles)
            jacobian_tip_position_wrt_alpha = J_angle_to_position[:,0].reshape((-1,1))
            jacobian_tip_position_wrt_beta = J_angle_to_position[:,1].reshape((-1,1))
            jacobian_tip_orientation_wrt_alpha = self.func_R_base_to_tip_derivative_wrt_alpha_(c_cables)
            jacobian_tip_orientation_wrt_beta = self.func_R_base_to_tip_derivative_wrt_beta_(c_cables)

            diff_P_in_Fc_wrt_alpha = jacobian_tip_orientation_wrt_alpha.T @ (P_in_Fb - tip_position) - tip_orientation.T @ jacobian_tip_position_wrt_alpha
            diff_P_in_Fc_wrt_beta = jacobian_tip_orientation_wrt_beta.T @ (P_in_Fb - tip_position) - tip_orientation.T @ jacobian_tip_position_wrt_beta

            # Compute jacobian of z wrt alpha and beta
            dz_wrt_alpha = np.array([
                        Kx*( diff_P_in_Fc_wrt_alpha[0]*P_in_Fc[2] - P_in_Fc[0]*diff_P_in_Fc_wrt_alpha[2]) /(P_in_Fc[2]**2),
                        Ky*( diff_P_in_Fc_wrt_alpha[1]*P_in_Fc[2] - P_in_Fc[1]*diff_P_in_Fc_wrt_alpha[2]) /(P_in_Fc[2]**2)
            ])
            dz_wrt_beta = np.array([
                        Kx*( diff_P_in_Fc_wrt_beta[0]*P_in_Fc[2] - P_in_Fc[0]*diff_P_in_Fc_wrt_beta[2])/(P_in_Fc[2]**2),
                        Ky*( diff_P_in_Fc_wrt_beta[1]*P_in_Fc[2] - P_in_Fc[1]*diff_P_in_Fc_wrt_beta[2])/(P_in_Fc[2]**2)
            ])
            jacobian_image_measurement_wrt_c[:,0] = dz_wrt_alpha.reshape((-1,))
            jacobian_image_measurement_wrt_c[:,1] = dz_wrt_beta.reshape((-1,))
            # Combine with jacobian of [alpha, beta] wrt c_cables
            J_cable_to_angle = self.func_jacobian_c_cables_to_c_angles_(c_cables)
            jacobian_image_measurement_wrt_c = jacobian_image_measurement_wrt_c @ J_cable_to_angle
            # 2) jacobian wrt p = [x, y, z}^T
            jacobian_image_measurement_wrt_p = np.array([
                [Kx/float(P_in_Fc[2]),  0.0,             -(Kx*float(P_in_Fc[0]))/(float(P_in_Fc[2])**2) ],
                [0.0,            Ky/float(P_in_Fc[2]),   -(Ky*float(P_in_Fc[1]))/(float(P_in_Fc[2])**2) ]
            ]) @ tip_orientation.T


        return jacobian_image_measurement_wrt_c, jacobian_image_measurement_wrt_p


    # Misc. utils (plotting)
    def get_wireframe_per_body(self, c_cables_) :
        c_cables = c_cables_.reshape((-1,1)).astype(float)
        c_angles = self.func_c_cables_to_c_angles_(c_cables)
        alpha = c_angles[0,0]
        beta = c_angles[1,0]
        # Draw base body (not controlled)
        pts_base = np.zeros((10,3))
        for idx in range(10):
            pts_base[idx, 2] = -0.05 *(9 - float(idx))/9
        # Draw bending section and tip
        pts_bending_section = np.zeros((20,3))
        pts_tip = np.zeros((10,3))
        if beta == 0 :
            for idx in range(pts_bending_section.shape[0]):
                pts_bending_section[idx, 2] = self.L_f_ * float(idx)/(pts_bending_section.shape[0]-1)
            for idx in range(pts_tip.shape[0]):
                pts_tip[idx, 2] = self.L_f_ + self.L_t_ * float(idx)/(pts_tip.shape[0]-1)
        else :
            # Draw bending section
            for idx in range(pts_bending_section.shape[0]):
                beta_tmp = beta * float(idx)/(pts_bending_section.shape[0]-1);
                tmp = (self.L_f_ / beta)*(1 - np.cos(beta_tmp));
                pts_bending_section[idx, 0] = np.cos(alpha)*tmp
                pts_bending_section[idx, 1] =  np.sin(alpha)*tmp
                pts_bending_section[idx, 2] = (self.L_f_ / beta)*np.sin(beta_tmp)
            # draw tip
            for idx in range(pts_tip.shape[0]):
                d_tmp = self.L_t_*(float(idx)/ (pts_tip.shape[0]-1));
                tmp = (self.L_f_ / beta)*(1 - np.cos(beta)) + d_tmp*np.sin(beta);
                pts_tip[idx, 0] = np.cos(alpha)*tmp
                pts_tip[idx, 1] =  np.sin(alpha)*tmp
                pts_tip[idx, 2] = (self.L_f_ / beta)*np.sin(beta) + d_tmp*np.cos(beta)

        return {
            "body" : pts_base,
            "flexible" : pts_bending_section,
            "tip" : pts_tip
        }
    def get_wireframe(self, c_cables) :
        wireframe_per_body = self.get_wireframe_per_body(c_cables)
        return np.concatenate([
                wireframe_per_body['body'],
                wireframe_per_body['flexible'],
                wireframe_per_body['tip']
            ])


#%%
if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    model = EndoscopeTwoBendingModel()

    def test_J_position(c) :
        print("\njacobian at (", float(c[0]) ,",", float(c[1]),") = \n",
              model.jacobian_position(c, use_approx = False))
        print("approx. jacobian at (", float(c[0]) ,",", float(c[1]),") = \n",
              model.jacobian_position(c, use_approx = True))
        print("err approx at (", float(c[0]) ,",", float(c[1]),") = \n",
              model.jacobian_position(c, use_approx = True) - model.jacobian_position(c, use_approx = False))

    test_singularity_R = model.R_base_to_tip(np.array([0,0]))
    test_singularity_t = model.t_base_to_tip(np.array([0,0]))
    test_J_position(np.array([0.,1e-5]))
    test_J_position(np.array([0.,-1e-5]))
    test_J_position(np.array([1e-5, 0.]))
    test_J_position(np.array([-1e-5, 0.]))

    def plot_motion(ax, q_values, label):
        p = np.zeros((100,3))
        for i in range(100) :
            #p[i,:] = model.t_base_to_tip(q_values[i,:]).reshape((-1,))
            p[i,:] = model.t_base_to_tip(q_values[i,:].reshape((-1,1))).reshape((-1,))
        return ax.plot(p[:,0], p[:,1], p[:,2], '-', marker='o', label=label)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.scatter(0,0,0, label="base")

    q_var_q1 = np.zeros((100,2))
    actuator_limit = np.pi/2 * model.D_
    q_var_q1[:,0] = np.linspace(-actuator_limit, actuator_limit, 100)
    q_var_q1[:,1] = 0


    q_var_q2 = np.zeros((100,2))
    q_var_q2[:,0] = 0 #np.linspace(-0.01, 0.01, 100)
    q_var_q2[:,1] = np.linspace(-actuator_limit, actuator_limit, 100)

    plot_motion(ax, q_var_q1, label='disp q1')
    plot_motion(ax, q_var_q2, label='disp q2 ')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
#%%
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(0,0,0, 'r',label="base")

    def plot_robot(ax, c) :
        #pts = model.get_wireframe(c)
        #ax.plot(pts[:,0], pts[:,1], pts[:,2], c=color, label=label)
        label="_dummy"
        wireframe_per_body = model.get_wireframe_per_body(c)
        ax.plot(wireframe_per_body['body'][:,0],
                wireframe_per_body['body'][:,1],
                wireframe_per_body['body'][:,2],
                '--',
                c='k', label=label)
        ax.plot(wireframe_per_body['flexible'][:,0],
                wireframe_per_body['flexible'][:,1],
                wireframe_per_body['flexible'][:,2],
                '--',
                c='g', label="_"+label)
        ax.plot(wireframe_per_body['tip'][:,0],
                wireframe_per_body['tip'][:,1],
                wireframe_per_body['tip'][:,2],
                '--',
                c='k', label="_"+label)
        ax.scatter(wireframe_per_body['tip'][-1,0],
                    wireframe_per_body['tip'][-1,1],
                    wireframe_per_body['tip'][-1,2],
                    c='r', label="_"+label)

    plot_robot(ax, np.array([0,0]))
    plot_robot(ax, np.array([0,0.005]))
    plot_robot(ax, np.array([0,-0.005]))
    plot_robot(ax, np.array([0.005,0]))
    plot_robot(ax, np.array([-0.005,0]))


    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    ax.set_box_aspect((np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    #%% try camera model
    from plot_ref_frames import plot_frame, draw_world_frame

    def plot_pose_3D(c, p, ax = None, ax_img = None) :
        if ax is None :
            fig = plt.figure()
            ax_img = fig.add_subplot(1,2,2)

            ax_img.set_xlim(model.camera_model_params_["u_lims"])
            ax_img.set_ylim(model.camera_model_params_["v_lims"])
            ax_img.set_aspect('equal')

            ax = fig.add_subplot(1,2,1, projection='3d')
            draw_world_frame(ax, 'Frame base')

        TR_44_Fc = np.eye(4)
        TR_44_Fc[:-1, -1] = model.t_base_to_tip(c).reshape((-1,))
        TR_44_Fc[:-1, :-1] = model.R_base_to_tip(c)
        plot_robot(ax, c)
        plot_frame(ax, TR_44 = TR_44_Fc, frame_name = 'Frame camera', arrows_length = 0.05)
        # Plot P
        ax.scatter(p[0], p[1], p[2], label="p_k")

        ax.set_xlabel('x_values')
        ax.set_ylabel('y_values')
        ax.set_zlabel('z_values')

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        ax.set_box_aspect((np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits)))
        ax.legend()

        # Image view
        if  p.reshape((-1,)).shape[0] == 3 :
            z = model.image_measurment_from_Fb(c, p)
            ax_img.scatter(z[0], z[1], label = "z_k")
        else :
            print("P shape not supported!")
        return ax, ax_img

    model.init_default_camera_model()
    #%%
    q_pos_eq = np.pi/4 * model.D_ / np.sqrt(2)

    c = np.array([-q_pos_eq, q_pos_eq])
    P = np.array([0.01,0,0.15]).reshape((-1,1))

    def test_at_pts(c,p, approx) :
        z = model.image_measurment_from_Fb(c, P)
        H_z_wrt_c, H_z_wrt_P_Fb = model.jacobian_image_measurement_wrt_c_and_P(c, P, approx)

        plot_pose_3D(c,P)
        print("z= \n", z)
        print("H_z_wrt_c = \n", H_z_wrt_c)
        print("H_z_wrt_P_Fb = \n", H_z_wrt_P_Fb)

    print('Analytical :')
    test_at_pts(c,P, approx=False)
    print('With approximation :')
    test_at_pts(c,P, approx=True)


    print('at (0,0), with approximation :')
    test_at_pts(np.array([0., 0.]),P, approx=True)

    ax, ax_img = plot_pose_3D(np.array([0., 0.]), P)
    plot_pose_3D(np.array([0., 0.001]), P, ax, ax_img)
    plot_pose_3D(np.array([0.001, 0.001]), P, ax, ax_img)
    plot_pose_3D(np.array([0.001, 0.]), P, ax, ax_img)
    plot_pose_3D(np.array([0.001, -0.001]), P, ax, ax_img)
    plot_pose_3D(np.array([0., -0.001]), P, ax, ax_img)
    plot_pose_3D(np.array([-0.001, -0.001]), P, ax, ax_img)
    plot_pose_3D(np.array([-0.001, 0.0]), P, ax, ax_img)
    plot_pose_3D(np.array([-0.001, 0.001]), P, ax, ax_img)



