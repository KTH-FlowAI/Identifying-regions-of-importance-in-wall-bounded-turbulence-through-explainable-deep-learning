# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:11:23 2023

@author: andres cremades botella

File for segmenting the domain in structures
"""

import get_data_fun as gd
import numpy as np

#%% Prepare data for training
start =  7000 #  
end =  7010 # 
delta = 1
normdata = gd.get_data_norm()
normdata.geom_param(start,1,1,1)
normdata.calc_Umean(start,end)
normdata.save_Umean()
#normdata.read_Umean()
normdata.plot_Umean()
normdata.calc_rms(start,end)
normdata.save_Urms()
#normdata.read_Urms()
normdata.plot_Urms()
normdata.calc_norm(start,end)
normdata.save_norm()
#normdata.read_norm
uv_struc = normdata.calc_uvstruc(start,end) 

normdata.eval_dz(start,end,1)
normdata.eval_volfilter(start,end,delta)
normdata.eval_filter(7000,7001,1) 

print('End: main_segment..py')
