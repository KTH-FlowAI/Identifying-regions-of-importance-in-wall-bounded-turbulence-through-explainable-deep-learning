# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
step = 1
shap = sc.shap_conf()
shap.read_data(step=step,absolute=True,testcases=True,numfield=-1,fieldini=0,readdata=False,saveuv=True)
shap.plot_shaps()
shap.plot_shaps_pdf(bin_num=20,lev_val=3.1)
shap.plot_shaps_uv()
shap.plot_shaps_pdf_probability(bin_num=20,lev_val=3.1)
shap.plot_shaps_uv_pdf_probability(bin_num=30,lev_val=3.1)
shap.plot_shaps_uv_pdf(bin_num=30,lev_val=3.1)
shap.plot_shaps_AR_pdf(bin_num=60,lev_val=3.1)
shap.plot_shaps_AR()
shap.plot_shaps_AR_scatter()
shap.plot_shaps_total()
shap.plot_shaps_total_noback()
shap.plot_shaps_kde()
