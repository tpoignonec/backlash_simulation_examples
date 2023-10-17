# Copyright 2023 ICUBE Laboratory, University of Strasbourg
# License: Apache License, Version 2.0
# Author: Thibault Poignonec (thibault.poignonec@gmail.com)

"""
Created on Mon Sep 26 09:17:33 2022

@author: Thiba
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add annotation `text` to an `Axes3d` instance.'''
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)
setattr(Axes3D, 'annotate3D', _annotate3D)

def rot_and_trans_to_TR44(R, t):
    _TR = block_diag(R, np.array([1]))
    _TR[0:3, 3] = t.reshape(-1)
    return _TR

def plot_frame(ax, TR_44, frame_name = '', arrows_length = .01, linewidth = 3) :
    _R = TR_44[0:3, 0:3]
    _t = TR_44[0:3,3]
    # Annotate frame
    ax.plot([_t[0]], [_t[1]], [_t[2]], 'o', markersize=10, color='red', alpha=0.5)

    ax.annotate3D(frame_name, (_t[0]-arrows_length/2, _t[1]-arrows_length/2, _t[2]-arrows_length/2),
                  xytext=(-arrows_length, -arrows_length),
                  textcoords='offset points',
    #              arrowprops=dict(ec='black', fc='white', shrink=2.5),
    )

    _base_vects = [_R[:,i] for i in range(3)]
    for _v, _c in zip(_base_vects, ['red', 'green', 'blue']):
        ax.quiver(
            _t[0], _t[1], _t[2],# <-- starting point of vector
            _v[0]*arrows_length, _v[1]*arrows_length, _v[2]*arrows_length, # <-- directions of vector
            color = _c, alpha = .8, lw = linewidth,
        )

def draw_world_frame(ax, label = "Frame world") :
    # Plot frame
    TR_origin = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    plot_frame(ax, TR_origin, label, arrows_length = 0.05)
    # Plot XY and  plane
    range_x_and_z = [-0.1, 0.2]
    span = abs(range_x_and_z[1] - range_x_and_z[0])
    range_y = [-span/2, span/2]
    _xx, _yy = np.meshgrid(np.linspace(range_x_and_z[0], range_x_and_z[1], 2),
                           np.linspace(range_y[0], range_y[1], 2))
    _zz = _xx
    #ax.plot_surface(_xx*0, _yy, _zz, alpha = 0.2)
    #ax.plot_surface(_xx, _yy, _zz*0, alpha = 0.2)
    #ax.plot_surface(_xx, _yy*0, _zz, alpha = 0.2)
