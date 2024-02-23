# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:11:23 2023

@author: andres cremades botella

File for segmenting the domain in structures
"""

import get_data_fun as gd
import numpy as np

#%% Prepare data for training
start =  1   
end =  11
normdata = gd.get_data_norm()
normdata.geom_param(start,1,1)
normdata.calc_rms_point()
normdata.save_Urms_point()
#normdata.read_Urms_point()
normdata.calc_rms()
normdata.save_Urms()
#normdata.read_Urms()
normdata.calc_norm()
normdata.save_norm()
#normdata.read_norm()
uv_struc = normdata.calc_uvstruc(Hperc=0.54)
Hperc = normdata.decideH(delta_field=100,eH_ini=-1,eH_fin=1,eH_delta=20)
fieldH = 11
normdata.plotsegmentation(fieldH,Hperc=1.75,filt=True)
normdata.Q_stat()
UU,VV = normdata.test_mean()
normdata.filter_struc(volfilt=2.7e4)
