# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

import numpy as np
from scipy.linalg import block_diag
#from copy import deepcopy

from .generic_backlash import AbstractBacklash

class MultiDofBacklash(AbstractBacklash):
    '''
    Multi-dof backlash component to model a transmission with multiple independent backlash sub-elements.
    Input actuator q in R^N
    Output c in R^N
    '''

    def __init__(self, ordered_components_list : list) :
        '''
        Instantiate a multi-dof backlash elements from a list of sub-models
        '''
        self._ordered_components_list = ordered_components_list #deepcopy(ordered_components_list)
        flatten = lambda l: [item for sublist in l for item in sublist]

        super().__init__(parameters_count = self.parameters_count,
                 parameters_labels = flatten(self.parameters_labels),
                 parameters_units = flatten(self.parameters_units),
                 q_label = [comp._q_label for comp in self._ordered_components_list],
                 q_unit = [comp._q_unit for comp in self._ordered_components_list],
                 c_label = [comp._c_label for comp in self._ordered_components_list],
                 c_unit = [comp._c_unit for comp in self._ordered_components_list],
        )

        self._model_name = "multi-dof backlash model"

        # Build indexes
        last_idx = 0
        self._x_i_indexes = []
        for component in self.ordered_components_list :
            size_x_i = int(component.x_dim)
            self._x_i_indexes.append(range(last_idx, last_idx+size_x_i))
            last_idx = last_idx+size_x_i

    # -----------------
    # Owned properties
    # -----------------
    @property
    def ordered_components_list(self):
        ''' List of backlash sub-companents, each inheriting from the 'GenericBacklash' class.'''
        return self._ordered_components_list

    @property
    def x_i_indexes(self) :
        ''' List of sub-components x_i indexes (as ranges) '''
        return self._x_i_indexes

    @property
    def c_i_indexes(self) :
        ''' List of sub-components c_i indexes (as idx) '''
        return [x_i_index[0] for x_i_index in self._x_i_indexes]

    # ------------------------------
    # Redefine inherited properties
    # ------------------------------
    @AbstractBacklash.parameters_labels.getter
    def parameters_labels(self) -> list:
        return [component.parameters_labels for component in self.ordered_components_list]

    @AbstractBacklash.parameters_units.getter
    def parameters_units(self) -> list:
        return [component.parameters_units for component in self.ordered_components_list]

    @AbstractBacklash.x_labels.getter
    def x_labels(self) -> list:
        return [component.x_labels for component in self.ordered_components_list]

    @AbstractBacklash.x_units.getter
    def x_units(self) -> list:
        return [component.x_units for component in self.ordered_components_list]

    @AbstractBacklash.x_dim.getter
    def x_dim(self) -> int :
        return sum([component.x_dim for component in self.ordered_components_list])

    @AbstractBacklash.parameters_count.getter
    def parameters_count(self) -> int:
        return sum([component.parameters_count for component in self.ordered_components_list])

    @AbstractBacklash.active_branch.getter
    def active_branch(self) -> list:
        active_branches = [component.active_branch for component in self.ordered_components_list]
        for component_active_branch in active_branches :
            if component_active_branch is None :
                raise Exception("'active_branch' not initialized! Have you called 'evaluate()'?")
        return active_branches

    # --------
    # Methods
    # --------
    def equivalent_backlash_width(self, x_k : np.ndarray) -> np.ndarray:
        x_k_ = x_k.reshape(-1)
        n_dof = len(self._ordered_components_list)
        B_widths = np.zeros((n_dof,))
        for idx_dof, x_i_index_range in zip(range(n_dof), self.x_i_indexes) :
            B_widths[idx_dof] = self._ordered_components_list[idx_dof].equivalent_backlash_width(x_k_[x_i_index_range])
        return B_widths

    def equivalent_mean_slope(self, x_k : np.ndarray) -> np.ndarray:
        x_k_ = x_k.reshape(-1)
        n_dof = len(self._ordered_components_list)
        mean_slopes = np.zeros((n_dof,))
        for idx_dof, x_i_index_range in zip(range(n_dof), self.x_i_indexes) :
            mean_slopes[idx_dof] = self._ordered_components_list[idx_dof].equivalent_mean_slope(x_k_[x_i_index_range])
        return mean_slopes

    def equivalent_mean_q_offset(self, x_k : np.ndarray) -> np.ndarray:
        x_k_ = x_k.reshape(-1)
        n_dof = len(self._ordered_components_list)
        mean_q_offsets = np.zeros((n_dof,))
        for idx_dof, x_i_index_range in zip(range(n_dof), self.x_i_indexes) :
            mean_q_offsets[idx_dof] = self._ordered_components_list[idx_dof].equivalent_mean_q_offset(x_k_[x_i_index_range])
        return mean_q_offsets


    def evaluate(self, x_k_1 : np.ndarray, q_k, q_k_1 = None) -> bool:
        n_dof = len(self._ordered_components_list)

        x_k_1_ = x_k_1.reshape(-1)
        q_k_ = q_k.reshape(-1)
        q_k_1_ = q_k_1.reshape(-1)

        x_k_ = np.zeros((self.x_dim,))*np.nan
        self._obs = np.zeros((self.x_dim,))*np.nan

        self._F = np.zeros((self.x_dim, self.x_dim))
        self._map_x_to_c = np.zeros((n_dof, self.x_dim))

        for idx_dof, x_i_index_range in zip(range(n_dof), self.x_i_indexes) :
            x_last_i = x_k_1_[x_i_index_range]
            q_i = q_k_[idx_dof]
            q_last_i = q_k_1_[idx_dof]
            x_k_[x_i_index_range] = self._ordered_components_list[idx_dof].evaluate(x_last_i, q_i, q_last_i)
            # Fill associated blocks in obs and map_c
            self._obs[x_i_index_range] = self._ordered_components_list[idx_dof].obs
            self._map_x_to_c[idx_dof, x_i_index_range] = self._ordered_components_list[idx_dof]._map_x_to_c
        # Build F
        F_list = [comp.F for comp in self._ordered_components_list]
        self._F = block_diag(*F_list)

        return x_k_

    def f(self, x_k_1, q_k) -> np.ndarray :
        x_k_1_ = x_k_1.reshape(-1)
        q_k_ = q_k.reshape((-1,))

        x_k_i_list = []
        for idx_dof, x_i_index_range in zip(range(len(self._ordered_components_list)), self.x_i_indexes) :
            x_last_i = x_k_1_[x_i_index_range]
            q_i = q_k_[idx_dof]
            x_k_i_list.append(self._ordered_components_list[idx_dof].f(x_last_i, q_i))
        return np.concatenate(x_k_i_list)

