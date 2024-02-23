# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
start = 7000
end = 7001
step = 1
shap = sc.shap_conf()
shap.calc_shap_kernel(start,end,step)
                      
print('End: main_shap.py')