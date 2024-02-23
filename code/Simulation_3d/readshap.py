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
shap.read_data(start,end,step,\
               absolute=True,readdata=False)
shap.plot_shaps(colormap='viridis')#
shap.plot_shaps_pdf(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_pdf_probability(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_pdf_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
shap.plot_shaps_pdf_wall_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
shap.plot_shaps_pdf_wall(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_uv(colormap='viridis')
shap.plot_shaps_uv_ejectionover(colormap='viridis')
shap.plot_shaps_uv_pdf(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_uv_pdf_probability(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_uv_pdf_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
shap.plot_shaps_uv_pdf_wall(colormap='viridis',bin_num=70,lev_val=5.1)
shap.plot_shaps_uv_pdf_wall_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
shap.plot_shaps_uv_pdf_hist(colormap='viridis')
shap.plot_shaps_kde(colormap='viridis')
shap.plot_shaps_AR(colormap='viridis')
shap.plot_shaps_total(colormap='viridis')
shap.plot_shaps_total_noback(colormap='viridis')


print('End: readshap.py')