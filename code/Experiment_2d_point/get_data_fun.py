# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:08:08 2023

@author: andres cremades botella

File containing the functions for reading the data
"""

import numpy as np
import h5py
import os

#%%
class get_data_norm():
    """
    Class for getting the normalization
    """
    
    def __init__(self,file_read='../../data/uv_fields_io/PIV',\
                 rey=1.377e+03,vtau=0.0414,pond='none'):
        """ 
        Initialize the normalization
        """
        self.file = file_read
        self.rey  = rey
        self.vtau = vtau
        self.pond = pond
        try:
            os.mkdir('../../results/Experiment_2d/')
        except:
            pass
        
    def geom_param(self,start,delta_y,delta_x,size_y=0.0348,size_x=0.1210):
        import glob
        self.delta_y = delta_y
        self.delta_x = delta_x
        file_ii = self.file+'.'+str(start)+'.*.h5.uvw'
        file_ii2 = glob.glob(file_ii)[0]
        file = h5py.File(file_ii2,'r+')  
        self.x = np.array(file['x'])[::delta_x,0] 
        self.y = np.array(file['y'])[0,::delta_y]
        self.my  = int((len(self.y)+delta_y-1)/delta_y)
        self.mx  = int((len(self.x)+delta_x-1)/delta_x)
        self.y_h = self.y/size_y
        self.x_h = self.x/size_y
        self.dy = np.zeros((self.my,))
        self.dx = np.zeros((self.mx,))
        self.yplus = self.y_h*self.rey
        self.xplus = self.x_h*self.rey
        for ii in np.arange(self.my):
            if ii==0:
                self.dy[ii] = abs(self.y_h[1]-self.y_h[0])/2
            elif ii==self.my-1:
                self.dy[ii] = abs(self.y_h[self.my-1]-self.y_h[self.my-2])/2
            else:
                self.dy[ii] = abs(self.y_h[ii+1]-self.y_h[ii-1])/2 
        for ii in np.arange(self.mx):
            if ii==0:
                self.dx[ii] = abs(self.x_h[1]-self.x_h[0])/2
            elif ii==self.mx-1:
                self.dx[ii] = abs(self.x_h[self.mx-1]-self.x_h[self.mx-2])/2
            else:
                self.dx[ii] = abs(self.x_h[ii+1]-self.x_h[ii-1])/2   
        self.vol = np.matmul(self.dx.reshape(self.mx,1),\
                             self.dy.reshape(1,self.my))*self.rey**2
        

    def test_mean(self):
        """
        function to ensure the mean is 0
        """
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)
        for ii in np.arange(len(listfiles)):
            try:
                file_ii = listfiles[ii]
                print('RMS velocity calculation:' + str(file_ii))
                file = h5py.File(folder+'/'+file_ii,'r+')
                flag = 1
            except:
                print('Reading failed...')
                flag = 0
            if flag == 1:
                uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
                vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
                if ii==0:
                    uu_mean = uu
                    vv_mean = vv
                else:
                    uu_mean += uu
                    vv_mean += vv
        uu_mean /= len(listfiles)
        vv_mean /= len(listfiles)
        return uu_mean,vv_mean

    def calc_rms(self):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations
        """
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)
        for ii in np.arange(len(listfiles)):
            try:
                file_ii = listfiles[ii]
                print('RMS velocity calculation:' + str(file_ii))
                file = h5py.File(folder+'/'+file_ii,'r+')
                flag = 1
            except:
                print('Reading failed...')
                flag = 0
            if flag == 1:
                uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
                vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
                uu2 = np.multiply(uu,uu)
                vv2 = np.multiply(vv,vv)
                uv  = np.multiply(uu,vv)
                if ii == 0:
                    uu2_cum = np.sum(uu2,axis=(0))
                    vv2_cum = np.sum(vv2,axis=(0))
                    uv_cum  = np.sum(uv,axis=(0))
                    nn_cum = np.ones((self.my,))*self.mx
                else:
                    uu2_cum += np.sum(uu2,axis=(0))
                    vv2_cum += np.sum(vv2,axis=(0))
                    uv_cum  += np.sum(uv,axis=(0))
                    nn_cum += np.ones((self.my,))*self.mx
        self.uurms = np.sqrt(np.divide(uu2_cum,nn_cum))    
        self.vvrms = np.sqrt(np.divide(vv2_cum,nn_cum))  
        self.uv    = np.divide(uv_cum,nn_cum)
        
        
    def calc_rms_point(self):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations
        """
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)
        for ii in np.arange(len(listfiles)):
            try:
                file_ii = listfiles[ii]
                print('RMS velocity calculation:' + str(file_ii))
                file = h5py.File(folder+'/'+file_ii,'r+')
                flag = 1
            except:
                print('Reading failed...')
                flag = 0
            if flag == 1:
                uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
                vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
                uu2 = np.multiply(uu,uu)
                vv2 = np.multiply(vv,vv)
                uv  = np.multiply(uu,vv)
                if ii == 0:
                    uu2_cum = uu2
                    vv2_cum = vv2
                    uv_cum  = uv
                    nn_cum  = 1
                else:
                    uu2_cum += uu2
                    vv2_cum += vv2
                    uv_cum  += uv
                    nn_cum += 1
        self.uurms_point = np.sqrt(uu2_cum/nn_cum)    
        self.vvrms_point = np.sqrt(vv2_cum/nn_cum)  
        self.uv_point    = uv_cum/nn_cum
        
    def plot_Urms(self):
        """
        Function to plot the rms velocity
        """
        import matplotlib.pyplot as plt
        try:
            os.mkdir('../../results/Experiment_2d/')
        except:
            pass
        
        # Divide Umean in the two semichannels
        uurms_plus = self.uurms/self.vtau
        vvrms_plus = self.vvrms/self.vtau
        uv_plus    = self.uv/self.vtau**2
         
        from matplotlib import cm  
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(self.yplus,uurms_plus,'-',color=cmap[0,:],label='PIV')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$u\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_u.png')
        plt.figure()
        plt.plot(self.yplus,vvrms_plus,'-',color=cmap[0,:],label='PIV')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/rms_v.png')
        
        plt.figure()
        plt.plot(self.yplus,uv_plus,'-',color=cmap[0,:],label='PIV')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([300,7500])
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uv.png')
        
        
    def save_Urms_point(self,file="../../results/Experiment_2d/Urms.h5"):
        """
        Function for saving the value of the rms velocity node by node
        """        
        hf = h5py.File(file, 'w')
        hf.create_dataset('urms', data=self.uurms_point)
        hf.create_dataset('vrms', data=self.vvrms_point)
        hf.create_dataset('uv', data=self.uv_point)
        
           
    def read_Urms_point(self,file="../../results/Experiment_2d/Urms.h5"):
        """
        Function for saving the value of the rms velocity node by node
        """        
        hf = h5py.File(file, 'r')
        self.uurms_point = np.array(hf['urms'])
        self.vvrms_point = np.array(hf['vrms'])
        self.uv_point = np.array(hf['uv'])
        
        
    def save_Urms(self,file="../../results/Experiment_2d/Urms.txt"):
        """
        Function for saving the value of the rms velocity
        """
        file_save = open(file, "w+")           
        content = str(self.uurms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vvrms.tolist())+'\n'
        file_save.write(content)           
        content = str(self.uv.tolist())+'\n'
        file_save.write(content)    
        
    def read_Urms(self,file="../../results/Experiment_2d/Urms.txt"):
        """
        Function for reading the rms velocity
        """
        file_read = open(file,"r")
        self.uurms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uv = np.array(file_read.readline().replace('[','').\
                           replace(']','').split(','),dtype='float')
     

    def read_velocity(self,ii,padpix=0,out=False): 
        """
        Function for read the velocity fluctuation
        """        
        import glob
        if out:
            file_ii = self.file+'.*.'+str(ii)+'.h5.uvw'
        else:
            file_ii = self.file+'.'+str(ii)+'.*.h5.uvw'
        file_ii2 = glob.glob(file_ii)[0]
        print('Normalization velocity calculation:' + str(file_ii2))
        file = h5py.File(file_ii2,'r+')    
        uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
        vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
        if padpix > 0 and out:
            uu_pad = uu[padpix:-padpix,:]
            vv_pad = vv[padpix:-padpix,:]
            uu = uu_pad.copy()
            vv = vv_pad.copy()
        return uu,vv
                
    def calc_norm(self,umeanfile="../../results/Experiment_2d/Umean.txt"):
        """
        Function for calculating the normalization of u, v, w
        """
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)
        for ii in np.arange(len(listfiles)):
            try:
                file_ii = listfiles[ii]
                print('Norm velocity calculation:' + str(file_ii))
                file = h5py.File(folder+'/'+file_ii,'r+')
                flag = 1
            except:
                print('Reading failed...')
                flag = 0
            if flag == 1:
                uu_i0 = np.array(file['U'])[::self.delta_x,::self.delta_y]
                vv_i0 = np.array(file['V'])[::self.delta_x,::self.delta_y]
                uv_i0  = np.multiply(uu_i0,vv_i0)
                if ii == 0:
                    self.uumax = np.max(uu_i0)
                    self.vvmax = np.max(vv_i0)
                    self.uumin = np.min(uu_i0)
                    self.vvmin = np.min(vv_i0)
                    self.uvmax = np.max(uv_i0)
                    self.uvmin = np.min(uv_i0)
                else:
                    self.uumax = np.max([self.uumax,np.max(uu_i0)])
                    self.vvmax = np.max([self.vvmax,np.max(vv_i0)])
                    self.uumin = np.min([self.uumin,np.min(uu_i0)])
                    self.vvmin = np.min([self.vvmin,np.min(vv_i0)])
                    self.uvmax = np.max([self.uvmax,np.max(uv_i0)])
                    self.uvmin = np.min([self.uvmin,np.min(uv_i0)])
                
    def save_norm(self,file="../../results/Experiment_2d/norm.txt"):
        """
        Function for saving the value of the normalization
        """
        file_save = open(file, "w+")           
        content = str(self.uumax)+'\n'
        file_save.write(content)    
        content = str(self.vvmax)+'\n'
        file_save.write(content)         
        content = str(self.uumin)+'\n'
        file_save.write(content)    
        content = str(self.vvmin)+'\n'
        file_save.write(content)         
        content = str(self.uvmax)+'\n'
        file_save.write(content)             
        content = str(self.uvmin)+'\n'
        file_save.write(content)       
                
    def read_norm(self,file="../../results/Experiment_2d/norm.txt"):
        """
        Function for reading the normalization
        """
        file_read = open(file,"r")
        self.uumax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uumin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uvmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uvmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
                
 
                
    def trainvali_data(self,index,ts=0.2,umeanfile="../../results/Experiment_2d/Umean.txt",\
                       normfile="../../results/Experiment_2d/norm.txt",delta_pred=1,padpix=0):
        
        from sklearn.model_selection import train_test_split        
        import tensorflow as tf
        vel_data_in = np.zeros((len(index),self.mx,self.my,2))
        vel_data_out = np.zeros((len(index),self.mx-2*padpix,self.my,2))
        try:
            self.uumin
            self.uumax
            self.vvmin
            self.vvmax
            self.wwmin
            self.wwmax
        except:
            self.read_norm(file=normfile)
        for ii in np.arange(len(index)): 
            ii_sub = 0
            flag = 1
            while flag == 1:
                try:
                    uu_i0,vv_i0 = self.read_velocity(index[ii]-ii_sub,padpix=padpix)
                    uu_i1,vv_i1 = self.read_velocity(index[ii]-ii_sub,padpix=padpix,out=True)
                    flag = 0
                except:
                    ii_sub += 1
            vel_data_in[ii,:,:,:] = self.norm_velocity(uu_i0,vv_i0)
            vel_data_out[ii,:,:,:] = self.norm_velocity(uu_i1,vv_i1)
        data_X = vel_data_in
        data_Y = vel_data_out
        
        train_X,valid_X,train_Y,valid_Y = \
        train_test_split(data_X, data_Y,test_size=ts,shuffle=False) 
        len_train = len(train_X[:,0,0,0])
        ind_val = index[len_train:].tolist()
        file_save = open('../../results/Experiment_2d/ind_val.txt', "w+")           
        content = str(ind_val)+'\n'
        file_save.write(content) 
        file_save.close()
        del data_X,data_Y
        train_X = train_X.reshape(-1,self.mx,self.my,2)
        valid_X = valid_X.reshape(-1,self.mx,self.my,2)
        train_Y = train_Y.reshape(-1,self.mx-2*padpix,self.my,2)
        valid_Y = valid_Y.reshape(-1,self.mx-2*padpix,self.my,2)
        train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
        del train_X,train_Y
        val_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))
        del valid_X,valid_Y
        return train_data,val_data
    
    def norm_velocity(self,uu,vv):
        """
        Function for reading a field given the index
        """
        vel_data = np.zeros((1,len(uu[:,0]),len(uu[0,:]),2))
        unorm = (uu-self.uumin)/(self.uumax-self.uumin)
        vnorm = (vv-self.vvmin)/(self.vvmax-self.vvmin)
        vel_data[0,:,:,0] = unorm
        vel_data[0,:,:,1] = vnorm
        return vel_data
    
        
    def dimensional_velocity(self,normfield,normfile="../../results/Experiment_2d/norm.txt"):
        """
        Function for transform the velocity to dimensional values
            * normfile : normalization file
        """
        velfield = np.zeros((len(normfield[0,:,0,0]),len(normfield[0,0,:,0]),2))
        try:
            velfield[:,:,0] = normfield[0,:,:,0]*\
            (self.uumax-self.uumin)+self.uumin
            velfield[:,:,1] = normfield[0,:,:,1]*\
            (self.vvmax-self.vvmin)+self.vvmin
        except:
            self.read_norm(file=normfile)
            velfield[:,:,0] = normfield[0,:,:,0]*\
            (self.uumax-self.uumin)+self.uumin
            velfield[:,:,1] = normfield[0,:,:,1]*\
            (self.vvmax-self.vvmin)+self.vvmin
        if self.pond == 'vel':
            velfield = np.sign(velfield)*np.sqrt(abs(velfield))
        return velfield
    

    def uvstruc_solve(self,file_field,urmsfile="../../results/Experiment_2d/Urms.h5",Hperc=1.75): 
        """
        Function for defining the Q structures in the domain
        """
        file = h5py.File(file_field,'r+')
        print('Calculating for:' + str(file_field)) 
        uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
        vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
        uv = abs(np.multiply(uu,vv))
        try:
            uvi = np.multiply(self.uurms,self.vvrms) 
        except:
            self.read_Urms_point(urmsfile)
            uvi = np.multiply(self.uurms_point,self.vvrms_point)  
        # Calculate where is the structure
        mat_struc = np.heaviside(uv-Hperc*uvi,0)
        # Calculate the structure properties
        uv_str = uvstruc(mat_struc)
        uv_str.get_cluster_3D6P()
        uv_str.get_volume_cluster_box(self.y_h,self.x_h,self.mx,self.vol)
        uv_str.geo_char(uu,vv,self.vol,self.mx,self.my)
        uv_str.filtstr_sum = 0
        uv_str.segmentation(self.mx,self.my)
        return uv_str
        
    def calc_uvstruc(self,delta_field=1,urmsfile="../../results/Experiment_2d/Urms.h5",\
                     Hperc=1.75,fileQ='../../results/Q_fields_io/PIV',\
                     fold='../../results/Q_fields_io'):
        """
        Function for calculating the uv structures
        """      
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)[::delta_field]
        number_cases = len(listfiles)
        for jj in np.arange(number_cases):
            file_jj = listfiles[jj]
            uv_str = self.uvstruc_solve(folder+'/'+file_jj,urmsfile=urmsfile,Hperc=Hperc)
            try:
                from os import mkdir
                mkdir(fold)
            except:
                pass
            index_piv = file_jj.find('PIV')
            fileQ_ii = fileQ+file_jj[index_piv+3:]
            fileQ_ii = fileQ_ii.replace('uvw','Q')
            hf = h5py.File(fileQ_ii, 'w')
            hf.create_dataset('Qs', data=uv_str.mat_struc)
            hf.create_dataset('Qs_event', data=uv_str.mat_event)
            hf.create_dataset('Qs_event_filtered', data=uv_str.mat_event_filtered)
            hf.create_dataset('Qs_segment', data=uv_str.mat_segment)
            hf.create_dataset('Qs_segment_filtered', data=uv_str.mat_segment_filtered)
            hf.create_dataset('dx', data=uv_str.dx)
            hf.create_dataset('ymin', data=uv_str.ymin)
            hf.create_dataset('ymax', data=uv_str.ymax)
            hf.create_dataset('vol', data=uv_str.vol)
            hf.create_dataset('volbox', data=uv_str.boxvol)
            hf.create_dataset('cdg_xbox', data=uv_str.cdg_xbox)
            hf.create_dataset('cdg_ybox', data=uv_str.cdg_ybox)
            hf.create_dataset('cdg_x', data=uv_str.cdg_x)
            hf.create_dataset('cdg_y', data=uv_str.cdg_y)
            hf.create_dataset('event', data=uv_str.event)
            hf.close()
        print('Percentage of filtered structures: '+\
              str(uv_str.filtstr_sum/number_cases*100)+'%')
            
        
 
        
    def read_uvstruc(self,fileQ_ii):
        """
        Function for reading the uv structures
        """
        uv_str = uvstruc()
        uv_str.read_struc(fileQ_ii)
        return uv_str
    
    def decideH(self,delta=1,out=False,eH_ini=-1,eH_fin=1,eH_delta=20,padpix=15,\
                fileQ='../../results/Q_fields_io/PIV',urmsfile="../../results/Experiment_2d/Urms.h5",colormap='viridis',
                delta_field=1,volfil=2.7e4):
        """
        Function for deciding the most appropriate H
        """
        import re
        try:
            os.mkdir('../../results/Experiment_2d/')
        except:
            pass
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)[::delta_field]
        number_cases = len(listfiles)
        struct_H = np.zeros((eH_delta,))
        struct_Hfilter = np.zeros((eH_delta,))
        volmax_struc = np.zeros((eH_delta,))
        for jj in np.arange(number_cases):
            file_jj = listfiles[jj]
            file = h5py.File(folder+'/'+file_jj,'r+')
            print('Calculating for:' + str(file_jj))
            eH_vec = np.linspace(eH_ini,eH_fin,eH_delta)
            H_vec = 10**eH_vec
            uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
            vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
            uv = abs(np.multiply(uu,vv))
            try:
                uvi = np.multiply(self.uurms,self.vvrms) 
            except:
                self.read_Urms_point(urmsfile)
                uvi = np.multiply(self.uurms_point,self.vvrms_point)
            for ii in np.arange(eH_delta):
                # Calculate where is the structure
                Hperc = H_vec[ii]
                mat_struc = np.heaviside(uv-Hperc*uvi,0)
                uv_str = uvstruc(mat_struc=mat_struc)
                uv_str.get_cluster_3D6P()
                uv_str.get_volume_cluster_box(self.y_h,self.x_h,self.mx,self.vol)
                index_filter = np.where(uv_str.vol>=volfil)[0]
                lenstruc = len(index_filter)
                if lenstruc == 0:
                    volrat = np.nan
                else:
                    volrat = np.max(uv_str.vol[index_filter])/np.sum(uv_str.vol[index_filter])
                struct_H[ii] += lenstruc
                struct_Hfilter[ii] += len(uv_str.vol)
                volmax_struc[ii] += volrat
        struct_H /= number_cases
        struct_Hfilter /= number_cases
        volmax_struc /= number_cases
        struct_H /= np.max(struct_H)
        struct_Hfilter /= np.max(struct_Hfilter)
        import matplotlib.pyplot as plt
        fig=plt.figure()
        fs = 20
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(H_vec,struct_H,color=plt.cm.get_cmap(colormap,4).colors[0,:],\
                 label='$N/N_{tot}$')
        plt.plot(H_vec,struct_Hfilter,color=plt.cm.get_cmap(colormap,4).colors[1,:],\
                 label='$N/N_{tot}$ (filtered)')
        plt.plot(H_vec,volmax_struc,color=plt.cm.get_cmap(colormap,4).colors[2,:],\
                 label='$V_{lar}/V_{tot}$')
        ax.set_xscale('log')
        plt.legend(fontsize=fs)
        plt.xlabel('$H$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/Nstruc_H.png')
        
    def plotsegmentation(self,fieldH,out=False,urmsfile="../../results/Experiment_2d/Urms.h5",Hperc=1.75,\
                         fileQ='../../results/Q_fields_io/PIV',colormap_Q='tab20',colormap_struc='viridis',filt=False):
        """
        Function for plotting the segmentation of the domain
        """
        import re
        import glob        
        try:
            os.mkdir('../../results/Experiment_2d/')
        except:
            pass
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        if out:
            file_ii = self.file+'.*.'+str(fieldH)+'.h5.uvw'
        else:
            file_ii = self.file+'.'+str(fieldH)+'.*.h5.uvw'
        file_ii2 = glob.glob(file_ii)[0]
        print('Plotting segmented field:' + str(file_ii2))
        uv_str = self.uvstruc_solve(file_ii2,urmsfile=urmsfile,Hperc=Hperc)        
        yy,xx = np.meshgrid(self.yplus,self.xplus)
        import matplotlib.pyplot as plt
        fs = 20
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        if filt:
            matfilt = uv_str.mat_segment_filtered
        else:
            matfilt = uv_str.mat_segment
        matfilt2 = matfilt.copy()
        matfilt[matfilt==0] = np.nan
        im0=axes.pcolor(xx,yy,matfilt,cmap=colormap_Q)
        axes.set_title(r"Field $\#$: "+str(fieldH)+', out: '+str(out)+' , H: '+str(Hperc),fontsize=fs)
        axes.set_ylabel('$y^+$',fontsize=fs)
        axes.set_xlabel('$x^+$',fontsize=fs)
        axes.tick_params(axis='both',which='major',labelsize=fs)
        axes.set_aspect('equal')
        axes.set_yticks([0,self.rey/2,self.rey])
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)  
        cb.set_label(r"$\# Q$",fontsize=fs)
        cb.ax.tick_params(labelsize=fs)
        try:
#            from os import mkdir
            os.mkdir('../../results/Experiment_2d/segment')
        except:
            pass
        plt.savefig('../../results/Experiment_2d/segment/seg_'+str(fieldH)+'_out_'+str(out)+'_H_'+str(Hperc)+'_filt_'+str(filt)+'.png')
        event = uv_str.mat_event
        event[event==0] = np.nan
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        im0=axes.pcolor(xx,yy,uv_str.mat_event,cmap=plt.cm.get_cmap(colormap_struc,4))
        axes.set_title(r"Field $\#$: "+str(fieldH)+', out: '+str(out)+' , H: '+str(Hperc),fontsize=fs)
        axes.set_ylabel('$y^+$',fontsize=fs)
        axes.set_xlabel('$x^+$',fontsize=fs)
        axes.tick_params(axis='both',which='major',labelsize=fs)
        axes.set_aspect('equal')
        axes.set_yticks([0,self.rey/2,self.rey])
        cbar = fig.colorbar(im0, ticks=[1,2,3,4])
        cbar.ax.set_yticklabels(['Outward\ninteraction','Ejection',\
                               'Inward\ninteraction','Sweep'],fontsize=fs-4)  
        try:
#            from os import mkdir
            os.mkdir('../../results/Experiment_2d/event')
        except:
            pass
        plt.savefig('../../results/Experiment_2d/event/seg_'+str(fieldH)+'_out_'+str(out)+'_H_'+str(Hperc)+'.png')
        
        file = h5py.File(file_ii2,'r+')
        uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
        vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        im0=axes.pcolor(xx,yy,uu/self.vtau,cmap=colormap_struc)
        im1=axes.contour(xx,yy,matfilt2,levels=np.max(matfilt2),colors='k')
        axes.set_title(r"Field $\#$: "+str(fieldH)+', out: '+str(out)+' , H: '+str(Hperc),fontsize=fs)
        axes.set_ylabel('$y^+$',fontsize=fs)
        axes.set_xlabel('$x^+$',fontsize=fs)
        axes.tick_params(axis='both',which='major',labelsize=fs)
        axes.set_aspect('equal')
        axes.set_yticks([0,self.rey/2,self.rey])
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)  
        cb.set_label(r"$u^+$",fontsize=fs)
        cb.ax.tick_params(labelsize=fs)  
        try:
#            from os import mkdir
            os.mkdir('../../results/Experiment_2d/vel')
        except:
            pass
        plt.savefig('../../results/Experiment_2d/vel/u_'+str(fieldH)+'_out_'+str(out)+'_H_'+str(Hperc)+'_filt_'+str(filt)+'.png')
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        im0=axes.pcolor(xx,yy,vv/self.vtau,cmap=colormap_struc)
        im1=axes.contour(xx,yy,matfilt2,levels=np.max(matfilt2),colors='k')
        axes.set_title(r"Field $\#$: "+str(fieldH)+', out: '+str(out)+' , H: '+str(Hperc),fontsize=fs)
        axes.set_ylabel('$y^+$',fontsize=fs)
        axes.set_xlabel('$x^+$',fontsize=fs)
        axes.tick_params(axis='both',which='major',labelsize=fs)
        axes.set_aspect('equal')
        axes.set_yticks([0,self.rey/2,self.rey])
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)  
        cb.set_label(r"$v^+$",fontsize=fs)
        cb.ax.tick_params(labelsize=fs)  
        try:
#            from os import mkdir
            os.mkdir('../../results/Experiment_2d/vel')
        except:
            pass
        plt.savefig('../../results/Experiment_2d/vel/v_'+str(fieldH)+'_out_'+str(out)+'_H_'+str(Hperc)+'_filt_'+str(filt)+'.png')
        uvrms = np.multiply(self.uurms_point,self.vvrms_point)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        im0=axes.pcolor(xx,yy,uvrms/self.vtau**2,cmap=colormap_struc)
        axes.set_title(r"Field $\#$: "+str(fieldH)+', out: '+str(out)+' , H: '+str(Hperc),fontsize=fs)
        axes.set_ylabel('$y^+$',fontsize=fs)
        axes.set_xlabel('$x^+$',fontsize=fs)
        axes.tick_params(axis='both',which='major',labelsize=fs)
        axes.set_aspect('equal')
        axes.set_yticks([0,self.rey/2,self.rey])
        cb = fig.colorbar(im0,orientation="vertical",aspect=20)
        cb.outline.set_visible(False)  
        cb.set_label(r"$(uv')^+$",fontsize=fs)
        cb.ax.tick_params(labelsize=fs)  
        try:
#            from os import mkdir
            os.mkdir('../../results/Experiment_2d/vel')
        except:
            pass
        plt.savefig('../../results/Experiment_2d/vel/urmsvrms_'+str(fieldH)+'_out_'+str(out)+'_H_'+str(Hperc)+'_filt_'+str(filt)+'.png')
        
  
    def filter_struc(self,delta=1,folder='../../results/Q_fields_io',padpix=0,volfilt=900):
        """
        Function for ploting Qs statistics
        """
        import os
        import re
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        listfiles = os.listdir(folder)[::delta]
        number_cases = len(listfiles)
        sum_filt = 0
        for ii in np.arange(number_cases):
            file_jj = listfiles[ii]
            uv_str = uvstruc()
            uv_str.read_struc(folder+'/'+file_jj)
            lenstruc = len(uv_str.vol)
            lenfilt = len(np.where(uv_str.vol<=volfilt)[0])
            sum_filt += lenfilt/lenstruc
        print('Percentage of filtered structures: '+str((sum_filt/number_cases)*100)+'%')
        
                
    def Q_stat(self,delta=1,folder='../../results/Q_fields_io',padpix=0):
        """
        Function for ploting Qs statistics
        """
        import re
        try:
            os.mkdir('../../results/Experiment_2d/')
        except:
            pass
        indexbar = [bar.start() for bar in re.finditer('/',self.file)]
        folder_uv = self.file[:indexbar[-1]]
        listfiles = os.listdir(folder)[::delta]
        number_cases = len(listfiles)
        nstruc = []
        nq1 = []
        nq2 = []
        nq3 = []
        nq4 = []
        volq1 = []
        volq2 = []
        volq3 = []
        volq4 = []
        uvq1 = []
        uvq2 = []
        uvq3 = []
        uvq4 = []
        for ii in np.arange(number_cases):
            file_jj = listfiles[ii]
            file_jjuv = file_jj.replace('Q','uvw')
            uv_str = uvstruc()
            uv_str.read_struc(folder+'/'+file_jj)
            lenstruc = len(uv_str.event)
            nstruc.append(lenstruc)
            q1ind = np.where(uv_str.event==1)[0]
            q2ind = np.where(uv_str.event==2)[0]
            q3ind = np.where(uv_str.event==3)[0]
            q4ind = np.where(uv_str.event==4)[0]
            vol1 = np.sum(uv_str.vol[q1ind])/np.sum(self.vol)
            vol2 = np.sum(uv_str.vol[q2ind])/np.sum(self.vol)
            vol3 = np.sum(uv_str.vol[q3ind])/np.sum(self.vol)
            vol4 = np.sum(uv_str.vol[q4ind])/np.sum(self.vol)
            file = h5py.File(folder_uv+'/'+file_jjuv,'r+')
            uu = np.array(file['U'])[::self.delta_x,::self.delta_y]
            vv = np.array(file['V'])[::self.delta_x,::self.delta_y]
            uvtot = np.sum(abs(np.multiply(uu,vv)))
            uv = np.zeros((lenstruc,))
            for jj in np.arange(lenstruc):
                indexuv = np.where(uv_str.mat_segment==jj+1)
                for kk in np.arange(len(indexuv[0])):
                    uv[jj] += abs(uu[indexuv[0][kk],indexuv[1][kk]]*vv[indexuv[0][kk],\
                                         indexuv[1][kk]]) 
            uv1 = np.sum(uv[q1ind])/uvtot
            uv2 = np.sum(uv[q2ind])/uvtot
            uv3 = np.sum(uv[q3ind])/uvtot
            uv4 = np.sum(uv[q4ind])/uvtot
            nq1.append(len(q1ind))
            nq2.append(len(q2ind))
            nq3.append(len(q3ind))
            nq4.append(len(q4ind))
            volq1.append(vol1)
            volq2.append(vol2)
            volq3.append(vol3)
            volq4.append(vol4)
            uvq1.append(uv1)
            uvq2.append(uv2)
            uvq3.append(uv3)
            uvq4.append(uv4)
            
        import matplotlib.pyplot as plt
        from matplotlib import cm  
        cmap = cm.get_cmap('viridis', 4).colors
        fs = 20
        cases_vec = np.arange(number_cases)
        plt.figure()
        plt.plot(cases_vec,nstruc,'-',color=cmap[0,:])
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$N$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/N_struc.png')
        plt.figure()
        plt.plot(cases_vec,nq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(cases_vec,nq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(cases_vec,nq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(cases_vec,nq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$N$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/N_Q_struc.png')
        plt.figure()
        plt.plot(cases_vec,volq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(cases_vec,volq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(cases_vec,volq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(cases_vec,volq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$V^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/V_Q_struc.png')
        plt.figure()
        plt.plot(cases_vec,uvq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(cases_vec,uvq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(cases_vec,uvq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(cases_vec,uvq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$uv$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Experiment_2d/uv_Q_struc.png')
            
        
        
class uvstruc():
    """
    Class containing the parameters of the uv structures
    """
    def __init__(self,mat_struc=[]):
        """ 
        initialization of the class
        """
        if len(mat_struc)>0:
            self.mat_struc = mat_struc
        try:
            os.mkdir('../../results/Q_fields_io/')
        except:
            pass
    
    def read_struc(self,fileQ_ii):
        file = h5py.File(fileQ_ii, 'r')
        print('Reading: '+fileQ_ii)
        mat_struc = np.array(file['Qs'])
        mat_event = np.array(file['Qs_event'])
        mat_event_filtered = np.array(file['Qs_event_filtered'])
        mat_segment = np.array(file['Qs_segment'])
        mat_segment_filtered = np.array(file['Qs_segment_filtered'])
        self.dx = np.array(file['dx'])
        self.ymin = np.array(file['ymin'])
        self.ymax = np.array(file['ymax'])
        self.vol = np.array(file['vol'])
        self.boxvol = np.array(file['volbox'])
        self.cdg_xbox = np.array(file['cdg_xbox'])
        self.cdg_ybox = np.array(file['cdg_ybox'])
        self.cdg_x = np.array(file['cdg_x'])
        self.cdg_y = np.array(file['cdg_y'])
        self.event = np.array(file['event'])
        self.mat_struc = mat_struc
        self.mat_event = mat_event
        self.mat_event_filtered = mat_event_filtered
        self.mat_segment = mat_segment
        self.mat_segment_filtered = mat_segment_filtered
        
        
    def get_cluster_3D6P(self):
        """
        Generate a sparse matrix to find the wall structures
        They are stored in a class.
        matrix_chong: can include all the domain or a part.
        """
        # Convert the structure matrix to int type, get its shape and copy
        wklocal  = self.mat_struc.astype('int')
        nx,ny = wklocal.shape
        wk       = wklocal.copy()
        # Create a matrix to evaluate the connectivity-1 in all the directions
        dirs = np.array([[-1,0],[0,-1],[1,0],[0,1]])
        # Create vectors with the index in the directions x,z,y note that the
        # directions with the symmetry: x,z add a vector with the index of the
        # nodes, the direction y adds a -1 to indicate the presence of the wall
        indx =  np.concatenate((np.array([-1]),np.arange(nx),np.array([-1])))
        indy = np.concatenate((np.array([-1]),np.arange(ny),np.array([-1]))) 
        # Create a vector infinitely large to store the nodes inside each 
        # structure
        pdim = 10**6      
        cola = np.zeros((2,5*pdim),dtype='int')
        nodes = np.zeros((2,pdim))
        self.nodes = []
        # Create a matrix of x,y,z for containing the index of the nodes in the
        # structures
        vrt = np.zeros((2,1),dtype='int')
        # Check the elements of the matrix
        for kk  in np.arange(0,ny):
            for ii  in np.arange(nx):
                # Skip the points where there is not a structure
                if wk[ii,kk] == 0:
                    continue
                # The first element of cola is composed by the nodes of
                # the element of the matrix used for calculating the 
                # connectivity
                cola[:,0] = np.array([ii,kk],dtype='int')
                # Initialization of the index
                nnp = 0
                nnq = 0
                ssq = 0
                while nnq <= ssq:
                    # initial point is taken from cola array, index nnq
                    # then the result is stored in nodes, index nnp
                    vrtini = cola[:,nnq]
                    nodes[:,nnp] = vrtini
                    # the index nnp is advanced and the studied element is
                    # removed from the matrix wk
                    nnp = nnp+1
                    wk[vrtini[0],vrtini[1]] = 0
                    # All the directions in the matrix dirs are checked
                    for ld in np.arange(4):
                        vrt[0] = indx[1+dirs[ld,0]+vrtini[0]]
                        vrt[1] = indy[1+dirs[ld,1]+vrtini[1]]
                        # calculate if the checked point is not a wall, 
                        # there is a structure or there was a point removed 
                        # before from the structure
                        while (not all(vrt[1]==-1)) and\
                        (not all(vrt[0]==-1)) and\
                        (wk[vrt[0],vrt[1]] == 1) and\
                        (wklocal[vrt[0],vrt[1]]==1):
                            # Advance the index of the nodes to store
                            # Delete the stored node and repeat
                            ssq = ssq+1
                            cola[:,ssq] = vrt.reshape(2)
                            wklocal[vrt[0],vrt[1]] = 0
                            vrt[0] = indx[1+dirs[ld,0]+vrt[0]]
                            vrt[1] = indy[1+dirs[ld,1]+vrt[1]]
                    nnq += 1
                # Define the nodes contained in the structure    
                self.nodes.append(nodes[:,:nnp].copy()) 
                    
    def get_volume_cluster_box(self,y_h,x_h,mx,vol):
        """
        Simplified function to calculate the approximate volume of a 3d cluster
        by calculating the volume of the containing box. The maximum and 
        minimum value are measured in every direction and multiplied  
        by the unitary distance between every point (hx,hz) and the y 
        which is not contant
        """
        # Create the information of the structures
        self.dx = np.zeros((len(self.nodes),))
        self.ymin = np.zeros((len(self.nodes),))
        self.ymax = np.zeros((len(self.nodes),))
        self.boxvol = np.zeros((len(self.nodes),))
        self.vol = np.zeros((len(self.nodes),))
        self.cdg_x = np.zeros((len(self.nodes),))
        self.cdg_y = np.zeros((len(self.nodes),))
        self.cdg_xbox = np.zeros((len(self.nodes),))
        self.cdg_ybox = np.zeros((len(self.nodes),))
        # Calculate for every structure
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            ymin = y_h[int(np.min(vpoints[1,:]))]
            ymax = y_h[int(np.max(vpoints[1,:]))]
            dy   = np.abs(ymax-ymin)
            x_sort = np.sort(vpoints[0,:])
            self.cdg_xbox[nn] = np.floor(np.mean(x_sort))
            self.cdg_ybox[nn] = np.floor(np.mean(vpoints[1,:])) 
            for nn2 in np.arange(len(self.nodes[nn][0,:])):
                self.cdg_x[nn] += x_h[vpoints[0,nn2]]*vol[vpoints[0,nn2],vpoints[1,nn2]]
                self.cdg_y[nn] += y_h[vpoints[1,nn2]]*vol[vpoints[0,nn2],vpoints[1,nn2]]
                self.vol[nn] += vol[vpoints[0,nn2],vpoints[1,nn2]]
            self.cdg_x[nn] /= self.vol[nn]
            self.cdg_y[nn] /= self.vol[nn]
            dx = x_h[np.max(x_sort)]-x_h[np.min(x_sort)] 
            self.dx[nn] = dx
            self.ymin[nn] = ymin
            self.ymax[nn] = ymax
            self.boxvol[nn] = dy*dx
                    
                    
    def geo_char(self,du,dv,vol,mx,my,filvol=900):
        """
        Function for calculating the geometrical characteristics of the uv 
        structures
        """
        # define the type of event matrix and the volume of each event
        self.mat_event = np.zeros((mx,my))
        self.mat_event_filtered = np.zeros((mx,my))
        self.event = np.zeros((len(self.nodes),))
        # Evaluate the characteristics for each structure
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            voltot = np.zeros((4,))
            # Evaluate each node of the structure
            for nn_node in np.arange(len(vpoints[0,:])):
                # get the u and v velocities for each point of the structure
                duval = du[vpoints[0,nn_node],vpoints[1,nn_node]]
                dvval = dv[vpoints[0,nn_node],vpoints[1,nn_node]]
                vol_nod = np.sqrt(duval**2+dvval**2)*\
                vol[vpoints[0,nn_node],vpoints[1,nn_node]]
                if duval > 0 and dvval > 0:
                    voltot[0] += vol_nod
                elif duval < 0 and dvval > 0:
                    voltot[1] += vol_nod
                elif duval < 0 and dvval < 0:
                    voltot[2] += vol_nod
                elif duval > 0 and dvval < 0:
                    voltot[3] += vol_nod
            max_event = np.argmax(voltot)
            if max_event == 0:
                self.event[nn] = 1
            elif max_event == 1:
                self.event[nn] = 2
            elif max_event == 2:
                self.event[nn] = 3
            elif max_event == 3:
                self.event[nn] = 4
            for nn_node in np.arange(len(vpoints[0,:])):
                self.mat_event[vpoints[0,nn_node],vpoints[1,nn_node]] = self.event[nn]
                if self.vol[nn] > filvol:
                    self.mat_event_filtered[vpoints[0,nn_node],vpoints[1,nn_node]] = self.event[nn]
                
                
    def segmentation(self,mx,my,filvol=900):
        """
        Function to segment the model
        """
        self.mat_segment = np.zeros((mx,my))
        self.mat_segment_filtered = np.zeros((mx,my))
        nn2 = 0
        nn3 = 0        
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            for nn_node in np.arange(len(vpoints[0,:])):
                self.mat_segment[vpoints[0,nn_node],vpoints[1,nn_node]] = nn+1
                if self.vol[nn] > filvol:
                    self.mat_segment_filtered[vpoints[0,nn_node],vpoints[1,nn_node]] = nn2+1                   
            if self.vol[nn] > filvol:
                nn2 += 1
            else:
                nn3 += 1
        self.filtstr_sum += nn3/(nn2+nn3)
        print('Percentage of filtered structures: '+str(nn3/(nn2+nn3)*100)+'%')

                    
                
        
        