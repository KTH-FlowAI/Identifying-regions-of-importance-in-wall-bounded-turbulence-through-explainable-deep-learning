# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
step = 1
shap = sc.shap_conf()
shap.calc_shap_deep(step=step,norep=False,testcases=True)
                      
#shap.eval_shap(step=step,testcases=True),model0='trained_model_0.h5'
