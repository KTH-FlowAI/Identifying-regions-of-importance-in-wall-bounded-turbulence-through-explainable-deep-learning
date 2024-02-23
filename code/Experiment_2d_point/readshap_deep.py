# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
step = 1
shap = sc.shap_conf()
shap.SHAP_rms(step=step,testcases=True,numfield=-1,fieldini=0)
shap.contourshap_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.segmentate_SHAP(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.contourQ_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.Qshap_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.find_SHAP_H(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.find_SHAP_H_abs(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.segmentateabs_SHAP(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.contourQabs_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.Qshapabs_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.plot_rms_shap(step=step,testcases=True,numfield=-1,fieldini=0)
shap.contourshapabs_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.Qoverlap_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)
shap.pierelativeabs_deep(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0)