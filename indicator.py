# -*- coding: utf-8 -*-
"""
indicator.py
Copyright (c) 2021 Nobuo Namura
This code is released under the MIT License.
"""

import numpy as np
from scipy.spatial import distance
from pygmo import hypervolume

#======================================================================
def rmse_history(x_rmse, problem, func, nfg=0):
    rmse = 0.0
    for x in x_rmse:
        rmse += (problem(x)[nfg] - func(x))**2.0
    rmse = np.sqrt(rmse/float(len(x_rmse[:,0])))
    return rmse

#======================================================================
def igd_history(f, igd_ref, plus=False, MIN=[]):
    if plus:
        if len(MIN) == 0:
            weight = np.full(len(f[0,:]), True)
        weight = - 1.0 + 2.0*MIN.astype(np.float)
        dist = weight*(np.tile(f, [len(igd_ref[:,0]),1,1]).transpose([1,0,2]) - np.tile(igd_ref, [len(f[:,0]),1,1]))
        dist = np.sqrt(np.sum(np.where(dist>0, dist, 0)**2, axis=2))
    else:
        dist = distance.cdist(f, igd_ref)
    igd = np.mean(np.min(dist,axis=0))
    return igd

#======================================================================
def hv_history(f, hv_ref, MIN=[]):
    f_hv = np.where(MIN, 1.0, -1.0)*(f - hv_ref)
    f_hv = f_hv[np.all(f_hv<0.0, axis=1),:]
    if len(f_hv) > 0:
        hv = hypervolume(f_hv).compute(np.zeros(len(MIN)))
    else:
        hv = 0.0
    return hv
