# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:06:29 2020

@author: GKSch
"""

import Vortex_Panel_Solver as vps 
import numpy as np

# Target design parameters
alpha_test_points = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]) * np.pi / 180.0
cl_test_points = np.array([0.2442, 0.4549, 0.7153, 0.9016, 1.0885, 1.2696])
cdp_test_points = np.array([0.00122, 0.00168, 0.00236, 0.00381, 0.00642, 0.00970])
cm4c_test_points = np.array([-0.0525, -0.0482, -0.0566, -0.0497, -0.0440, -0.0378])

# Simulation parameters
n_panel_per_surface = 14
n_steps = int(12.25 * (2*n_panel_per_surface + 1)) # In this number of steps, all vertices can be moved from min to max value
    
env = vps.Vortex_Panel_Solver(n_steps, n_panel_per_surface, alpha_test_points, cl_test_points, 
                              cdp_test_points, cm4c_test_points, symmetric=True, debug=True)