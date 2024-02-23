# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import shap_config as sc
step = 1
shap = sc.shap_conf()
shap.calc_shap_kernel(step=step,norep=True,testcases=True,numfield=-1,fieldini=0)
                      
