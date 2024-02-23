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
    
    def __init__(self,file_read='../../data/P125_21pi_vu/P125_21pi_vu',\
                 rey=125,vtau=0.060523258443963,pond='none'):
        """ 
        Initialize the normalization
        """
        self.file = file_read
        self.rey  = rey
        self.vtau = vtau
        self.pond = pond
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass
        
    def geom_param(self,start,delta_y,delta_z,delta_x,size_x=2*np.pi,\
                   size_z=np.pi):
        self.delta_y = delta_y
        self.delta_x = delta_x
        self.delta_z = delta_z
        file_ii = self.file+'.'+str(start)+'.h5.uvw'
        file = h5py.File(file_ii,'r+')        
        self.my  = int((np.array(file['my'])[0]+delta_y-1)/delta_y)
        self.mx  = int((np.array(file['mx'])[0]+delta_x-1)/delta_x)
        self.mz  = int((np.array(file['mz'])[0]+delta_z-1)/delta_z)
        self.y_h = np.array(file['y'])[::delta_y]
        self.dy = np.zeros((self.my,))
        if np.mod(self.my,2) == 0:
            self.yd_s = int(self.my*0.5)
            self.yu_s = int(self.my*0.5)
        else:
            self.yd_s = int(self.my*0.5)+1
            self.yu_s = int(self.my*0.5)
        self.yplus = (1+self.y_h[:self.yd_s])*self.rey
        self.xplus = np.linspace(0,size_x,self.mx)*self.rey
        self.zplus = np.linspace(0,size_z,self.mz)*self.rey
        for ii in np.arange(self.my):
            if ii==0:
                self.dy[ii] = (self.y_h[1]-self.y_h[0])/2
            elif ii==self.my-1:
                self.dy[ii] = (self.y_h[self.my-1]-self.y_h[self.my-2])/2
            else:
                self.dy[ii] = (self.y_h[ii+1]-self.y_h[ii-1])/2  
        self.dx = size_x/self.mx
        self.dz = size_z/self.mz
        vol_vec = self.dy*self.dx*self.dz*self.rey**3
        self.vol = np.zeros((1,self.mz,self.mx),dtype=vol_vec.dtype)+\
        vol_vec.reshape(-1,1,1)
        self.voltot = size_x*size_z*2*self.rey**3
        

        
    def calc_Umean(self,start,end):
        """
        Function to calculate the mean velocity
        """       
        for ii in range(start,end):            
            file_ii = self.file+'.'+str(ii)+'.h5.uvw'
            print('Mean velocity calculation:' + str(file_ii))
            file = h5py.File(file_ii,'r+')
            UU = np.array(file['u'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            VV = np.array(file['v'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            WW = np.array(file['w'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            if ii == start:
                UU_cum = np.sum(UU,axis=(1,2))
                VV_cum = np.sum(VV,axis=(1,2))
                WW_cum = np.sum(WW,axis=(1,2))
                nn_cum = np.ones((self.my,))*self.mx*self.mz
            else:
                UU_cum += np.sum(UU,axis=(1,2))
                VV_cum += np.sum(VV,axis=(1,2))
                WW_cum += np.sum(WW,axis=(1,2))
                nn_cum += np.ones((self.my,))*self.mx*self.mz
        self.UUmean = np.divide(UU_cum,nn_cum)
        self.VVmean = np.divide(VV_cum,nn_cum)
        self.WWmean = np.divide(WW_cum,nn_cum)
        
    def plot_Umean(self,reference='../../data/Simulation_3d/Re180.prof.txt'):
        """
        Function to plot the mean velocity
            * reference: file from the torroja ddbb
        """
        import matplotlib.pyplot as plt
        posy = []
        U = []
        with open(reference) as f:
            line = f.readline()
            while line:
                if line[0] != '%':
                    linesep = line.split()
                    posy.append(float(linesep[1]))
                    U.append(float(linesep[2]))
                line = f.readline()
        posy_arr = np.array(posy)
        U_arr = np.array(U)
        
        # Divide Umean in the two semichannels
        UUmean_dplus = self.UUmean[:self.yd_s]/self.vtau
        UUmean_uplus = np.flip(self.UUmean[self.yu_s:])/self.vtau
        VVmean_dplus = self.VVmean[:self.yd_s]/self.vtau
        VVmean_uplus = np.flip(self.VVmean[self.yu_s:])/self.vtau
        WWmean_dplus = self.WWmean[:self.yd_s]/self.vtau
        WWmean_uplus = np.flip(self.WWmean[self.yu_s:])/self.vtau
            
        from matplotlib import cm
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass  
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(self.yplus,UUmean_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,UUmean_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,U_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\overline{U}^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/mean_U.png')
        plt.figure()
        plt.plot(self.yplus,VVmean_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,VVmean_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\overline{V}^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/mean_V.png')
        plt.figure()
        plt.plot(self.yplus,WWmean_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,WWmean_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$\overline{W}^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/mean_W.png')
    
    def save_Umean(self,file="../../results/Simulation_3d/Umean.txt"):
        """
        Function for saving the value of the mean velocity
        """
        file_save = open(file, "w+")           
        content = str(self.UUmean.tolist())+'\n'
        file_save.write(content)          
        content = str(self.VVmean.tolist())+'\n'
        file_save.write(content)          
        content = str(self.WWmean.tolist())+'\n'
        file_save.write(content)
        
    def read_Umean(self,file="../../results/Simulation_3d/Umean.txt",delta_y=1,delta_z=1,delta_x=1):
        """
        Function for reading the mean velocity
        """
        file_read = open(file,"r")
        self.UUmean = np.array(file_read.readline().replace('[','').\
                         replace(']','').split(','),dtype='float')[::self.delta_y]
        self.VVmean = np.array(file_read.readline().replace('[','').\
                         replace(']','').split(','),dtype='float')[::self.delta_y]
        self.WWmean = np.array(file_read.readline().replace('[','').\
                         replace(']','').split(','),dtype='float')[::self.delta_y]
    
    def calc_rms(self,start,end,umeanfile="../../results/Simulation_3d/Umean.txt"):
        """
        Function for calculating the rms of the velocity components and the 
        product of the velocity fluctuations
        """
        for ii in range(start,end):
            file_ii = self.file+'.'+str(ii)+'.h5.uvw'
            print('RMS velocity calculation:' + str(file_ii))
            file = h5py.File(file_ii,'r+')
            UU = np.array(file['u'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            try:
                uu = UU-self.UUmean.reshape(-1,1,1)
            except:
                self.read_Umean(umeanfile)
                uu = UU-self.UUmean.reshape(-1,1,1)
            vv = np.array(file['v'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            ww = np.array(file['w'])[::self.delta_y,\
                         ::self.delta_z,::self.delta_x]
            uu2 = np.multiply(uu,uu)
            vv2 = np.multiply(vv,vv)
            ww2 = np.multiply(ww,ww)
            uv  = np.multiply(uu,vv)
            vw  = np.multiply(vv,ww)
            uw  = np.multiply(uu,ww)
            if ii == start:
                uu2_cum = np.sum(uu2,axis=(1,2))
                vv2_cum = np.sum(vv2,axis=(1,2))
                ww2_cum = np.sum(ww2,axis=(1,2))
                uv_cum  = np.sum(uv,axis=(1,2))
                vw_cum  = np.sum(vw,axis=(1,2))
                uw_cum  = np.sum(uw,axis=(1,2))
                nn_cum = np.ones((self.my,))*self.mx*self.mz
            else:
                uu2_cum += np.sum(uu2,axis=(1,2))
                vv2_cum += np.sum(vv2,axis=(1,2))
                ww2_cum += np.sum(ww2,axis=(1,2))
                uv_cum  += np.sum(uv,axis=(1,2))
                vw_cum  += np.sum(vw,axis=(1,2))
                uw_cum  += np.sum(uw,axis=(1,2))
                nn_cum += np.ones((self.my,))*self.mx*self.mz
        self.uurms = np.sqrt(np.divide(uu2_cum,nn_cum))    
        self.vvrms = np.sqrt(np.divide(vv2_cum,nn_cum))   
        self.wwrms = np.sqrt(np.divide(ww2_cum,nn_cum)) 
        self.uv    = np.divide(uv_cum,nn_cum)
        self.vw    = np.divide(vw_cum,nn_cum)
        self.uw    = np.divide(uw_cum,nn_cum)
        
        
    def plot_Urms(self,reference='../../data/Simulation_3d/Re180.prof.txt'):
        """
        Function to plot the rms velocity
            * reference: file from the torroja ddbb
        """
        import matplotlib.pyplot as plt
        posy = []
        u = []
        v = []
        w = []
        uv = []
        vw = []
        uw = []
        with open(reference) as f:
            line = f.readline()
            while line:
                if line[0] != '%':
                    linesep = line.split()
                    posy.append(float(linesep[1]))
                    u.append(float(linesep[3]))
                    v.append(float(linesep[4]))
                    w.append(float(linesep[5]))
                    uv.append(float(linesep[10]))
                    vw.append(float(linesep[12]))
                    uw.append(float(linesep[11]))
                line = f.readline()
        posy_arr = np.array(posy)
        u_arr = np.array(u)
        v_arr = np.array(v)
        w_arr = np.array(w)
        uv_arr = np.array(uv)
        vw_arr = np.array(vw)
        uw_arr = np.array(uw)
        
        # Divide Umean in the two semichannels
        uurms_dplus = self.uurms[:self.yd_s]/self.vtau
        vvrms_dplus = self.vvrms[:self.yd_s]/self.vtau
        wwrms_dplus = self.wwrms[:self.yd_s]/self.vtau
        uurms_uplus = np.flip(self.uurms[self.yu_s:])/self.vtau
        vvrms_uplus = np.flip(self.vvrms[self.yu_s:])/self.vtau
        wwrms_uplus = np.flip(self.wwrms[self.yu_s:])/self.vtau
        uv_dplus    = self.uv[:self.yd_s]/self.vtau**2
        vw_dplus    = self.vw[:self.yd_s]/self.vtau**2
        uw_dplus    = self.uw[:self.yd_s]/self.vtau**2
        uv_uplus    = -np.flip(self.uv[self.yu_s:])/self.vtau**2
        vw_uplus    = -np.flip(self.vw[self.yu_s:])/self.vtau**2
        uw_uplus    = np.flip(self.uw[self.yu_s:])/self.vtau**2
         
        from matplotlib import cm 
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass 
        cmap = cm.get_cmap('viridis', 5).colors
        fs = 20
        plt.figure()
        plt.plot(self.yplus,uurms_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,uurms_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,u_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$u\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_u.png')
        plt.figure()
        plt.plot(self.yplus,vvrms_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,vvrms_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,v_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$v\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_v.png')
        plt.figure()
        plt.plot(self.yplus,wwrms_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,wwrms_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,w_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$w\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/rms_w.png')
        
        plt.figure()
        plt.plot(self.yplus,uv_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,uv_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,uv_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uv\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uv.png')
        plt.figure()
        plt.plot(self.yplus,vw_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,vw_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,vw_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$vw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/vw.png')
        plt.figure()
        plt.plot(self.yplus,uw_dplus,'-',color=cmap[0,:],label='DNS lower')
        plt.plot(self.yplus,uw_uplus,'--',color=cmap[0,:],label='DNS upper')
        plt.plot(posy_arr,uw_arr,'-',color=cmap[3,:],label='torroja')
        plt.xlabel('$y^+$',fontsize=fs)
        plt.ylabel('$uw\'^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.xscale('log')
        plt.legend(fontsize=fs)
        plt.xlim([1,125])
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uw.png')
        
        
    def save_Urms(self,file="../../results/Simulation_3d/Urms.txt"):
        """
        Function for saving the value of the rms velocity
        """
        file_save = open(file, "w+")           
        content = str(self.uurms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vvrms.tolist())+'\n'
        file_save.write(content)    
        content = str(self.wwrms.tolist())+'\n'
        file_save.write(content)          
        content = str(self.uv.tolist())+'\n'
        file_save.write(content)    
        content = str(self.vw.tolist())+'\n'
        file_save.write(content)    
        content = str(self.uw.tolist())+'\n'
        file_save.write(content)
        
    def read_Urms(self,file="../../results/Simulation_3d/Urms.txt"):
        """
        Function for reading the rms velocity
        """
        file_read = open(file,"r")
        self.uurms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.wwrms = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uv = np.array(file_read.readline().replace('[','').\
                           replace(']','').split(','),dtype='float')
        self.vw = np.array(file_read.readline().replace('[','').\
                           replace(']','').split(','),dtype='float')
        self.uw = np.array(file_read.readline().replace('[','').\
                           replace(']','').split(','),dtype='float')
     

    def read_velocity(self,ii,padpix=0): 
        """
        Function for read the velocity fluctuation
        """        
        file_ii = self.file+'.'+str(ii)+'.h5.uvw'
        print('Normalization velocity calculation:' + str(file_ii))
        file = h5py.File(file_ii,'r+')    
        UU = np.array(file['u'])[::self.delta_y,\
                     ::self.delta_z,::self.delta_x]
        uu = UU-self.UUmean.reshape(-1,1,1)
        vv = np.array(file['v'])[::self.delta_y,\
                     ::self.delta_z,::self.delta_x]
        ww = np.array(file['w'])[::self.delta_y,\
                     ::self.delta_z,::self.delta_x] 
        if padpix > 0:
            fshape = uu.shape
            dim0 = fshape[0]
            dim1 = fshape[1]
            dim2 = fshape[2]
            uu_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            vv_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            ww_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            uu_pad[:,padpix:-padpix,padpix:-padpix] = uu.copy()
            uu_pad[:,:padpix,padpix:-padpix] = uu[:,-padpix:,:]
            uu_pad[:,-padpix:,padpix:-padpix] = uu[:,:padpix,:]
            uu_pad[:,:,:padpix] = uu_pad[:,:,-2*padpix:-padpix]
            uu_pad[:,:,-padpix:] = uu_pad[:,:,padpix:2*padpix]
            vv_pad[:,padpix:-padpix,padpix:-padpix] = vv.copy()
            vv_pad[:,:padpix,padpix:-padpix] = vv[:,-padpix:,:]
            vv_pad[:,-padpix:,padpix:-padpix] = vv[:,:padpix,:]
            vv_pad[:,:,:padpix] = vv_pad[:,:,-2*padpix:-padpix]
            vv_pad[:,:,-padpix:] = vv_pad[:,:,padpix:2*padpix]
            ww_pad[:,padpix:-padpix,padpix:-padpix] = ww.copy()
            ww_pad[:,:padpix,padpix:-padpix] = ww[:,-padpix:,:]
            ww_pad[:,-padpix:,padpix:-padpix] = ww[:,:padpix,:]
            ww_pad[:,:,:padpix] = ww_pad[:,:,-2*padpix:-padpix]
            ww_pad[:,:,-padpix:] = ww_pad[:,:,padpix:2*padpix]
            uu = uu_pad.copy()
            vv = vv_pad.copy()
            ww = ww_pad.copy()
        return uu,vv,ww
                
    def calc_norm(self,start,end,umeanfile="../../results/Simulation_3d/Umean.txt"):
        """
        Function for calculating the normalization of u, v, w
        """
        try:
            self.UUmean 
        except:
            try:
                self.read_Umean(umeanfile)
            except:
                self.calc_Umean(start,end)
        for ii in range(start,end):
            uu_i0,vv_i0,ww_i0 = self.read_velocity(ii)
            uu_i1,vv_i1,ww_i1 = self.read_velocity(ii+1)
            uv_i0 = np.multiply(uu_i0,vv_i0)
            vw_i0 = np.multiply(vv_i0,ww_i0)
            uw_i0 = np.multiply(uu_i0,ww_i0)
            if ii == start:
                self.uumax = np.max(uu_i0)
                self.vvmax = np.max(vv_i0)
                self.wwmax = np.max(ww_i0)
                self.uumin = np.min(uu_i0)
                self.vvmin = np.min(vv_i0)
                self.wwmin = np.min(ww_i0)
                self.uvmax = np.max(uv_i0)
                self.vwmax = np.max(vw_i0)
                self.uwmax = np.max(uw_i0)
                self.uvmin = np.min(uv_i0)
                self.vwmin = np.min(vw_i0)
                self.uwmin = np.min(uw_i0)
            else:
                self.uumax = np.max([self.uumax,np.max(uu_i0)])
                self.vvmax = np.max([self.vvmax,np.max(vv_i0)])
                self.wwmax = np.max([self.wwmax,np.max(ww_i0)])
                self.uumin = np.min([self.uumin,np.min(uu_i0)])
                self.vvmin = np.min([self.vvmin,np.min(vv_i0)])
                self.wwmin = np.min([self.wwmin,np.min(ww_i0)])
                self.uvmax = np.max([self.uvmax,np.max(uv_i0)])
                self.vwmax = np.max([self.vwmax,np.max(vw_i0)])
                self.uwmax = np.max([self.uwmax,np.max(uw_i0)])
                self.uvmin = np.min([self.uvmin,np.min(uv_i0)])
                self.vwmin = np.min([self.vwmin,np.min(vw_i0)])
                self.uwmin = np.min([self.uwmin,np.min(uw_i0)])
                
    def save_norm(self,file="../../results/Simulation_3d/norm.txt"):
        """
        Function for saving the value of the normalization
        """
        file_save = open(file, "w+")           
        content = str(self.uumax)+'\n'
        file_save.write(content)    
        content = str(self.vvmax)+'\n'
        file_save.write(content)    
        content = str(self.wwmax)+'\n'
        file_save.write(content)          
        content = str(self.uumin)+'\n'
        file_save.write(content)    
        content = str(self.vvmin)+'\n'
        file_save.write(content)    
        content = str(self.wwmin)+'\n'
        file_save.write(content)         
        content = str(self.uvmax)+'\n'
        file_save.write(content)    
        content = str(self.vwmax)+'\n'
        file_save.write(content)    
        content = str(self.uwmax)+'\n'
        file_save.write(content)          
        content = str(self.uvmin)+'\n'
        file_save.write(content)    
        content = str(self.vwmin)+'\n'
        file_save.write(content)    
        content = str(self.uwmin)+'\n'
        file_save.write(content)         
#        content = str(self.uudmax)+'\n'
#        file_save.write(content)    
#        content = str(self.vvdmax)+'\n'
#        file_save.write(content)    
#        content = str(self.wwdmax)+'\n'
#        file_save.write(content)          
#        content = str(self.uudmin)+'\n'
#        file_save.write(content)    
#        content = str(self.vvdmin)+'\n'
#        file_save.write(content)    
#        content = str(self.wwdmin)+'\n'
#        file_save.write(content) 
                
    def read_norm(self,file="../../results/Simulation_3d/norm.txt"):
        """
        Function for reading the normalization
        """
        file_read = open(file,"r")
        self.uumax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.wwmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uumin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vvmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.wwmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uvmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vwmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uwmax = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uvmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.vwmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
        self.uwmin = np.array(file_read.readline().replace('[','').\
                              replace(']','').split(','),dtype='float')
                
    
                
    def trainvali_data(self,index,ts=0.2,umeanfile="../../results/Simulation_3d/Umean.txt",\
                       normfile="../../results/Simulation_3d/norm.txt",delta_pred=1,padpix=0):
        
        from sklearn.model_selection import train_test_split        
        import tensorflow as tf
        vel_data_in = np.zeros((len(index),self.my,self.mz+2*padpix,\
                                self.mx+2*padpix,3))
        vel_data_out = np.zeros((len(index),self.my,self.mz,self.mx,3))
        try:
            self.UUmean 
        except:
            self.read_Umean(umeanfile)
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
            uu_i0,vv_i0,ww_i0 = self.read_velocity(index[ii],padpix=padpix)
            vel_data_in[ii,:,:,:,:] = self.norm_velocity(uu_i0,vv_i0,ww_i0,\
                       padpix=padpix)
            uu_i1,vv_i1,ww_i1 = self.read_velocity(index[ii]+delta_pred)
            vel_data_out[ii,:,:,:,:] = self.norm_velocity(uu_i1,vv_i1,ww_i1)
        data_X = vel_data_in
        data_Y = vel_data_out
        train_X,valid_X,train_Y,valid_Y = \
        train_test_split(data_X, data_Y,test_size=ts,shuffle=False,\
                         random_state=13) 
        del data_X,data_Y
        train_X = train_X.reshape(-1,self.my,self.mz+2*padpix,self.mx+2*padpix,3)
        valid_X = valid_X.reshape(-1,self.my,self.mz+2*padpix,self.mx+2*padpix,3) 
        train_Y = train_Y.reshape(-1,self.my,self.mz,self.mx,3)
        valid_Y = valid_Y.reshape(-1,self.my,self.mz,self.mx,3)
        train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
        del train_X,train_Y
        val_data = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))
        del valid_X,valid_Y
        return train_data,val_data
    
    def norm_velocity(self,uu,vv,ww,padpix=0):
        """
        Function for reading a field given the index
        """
        vel_data = np.zeros((1,self.my,self.mz+2*padpix,self.mx+2*padpix,3))
        unorm = (uu-self.uumin)/(self.uumax-self.uumin)
        vnorm = (vv-self.vvmin)/(self.vvmax-self.vvmin)
        wnorm = (ww-self.wwmin)/(self.wwmax-self.wwmin)
        vel_data[0,:,:,:,0] = unorm
        vel_data[0,:,:,:,1] = vnorm
        vel_data[0,:,:,:,2] = wnorm
        return vel_data
    
        
    def dimensional_velocity(self,normfield,normfile="../../results/Simulation_3d/norm.txt"):
        """
        Function for transform the velocity to dimensional values
            * normfile : normalization file
        """
        velfield = np.zeros((self.my,self.mz,self.mx,3))
        try:
            velfield[:,:,:,0] = normfield[0,:,:,:,0]*\
            (self.uumax-self.uumin)+self.uumin
            velfield[:,:,:,1] = normfield[0,:,:,:,1]*\
            (self.vvmax-self.vvmin)+self.vvmin
            velfield[:,:,:,2] = normfield[0,:,:,:,2]*\
            (self.wwmax-self.wwmin)+self.wwmin
        except:
            self.read_norm(file=normfile)
            velfield[:,:,:,0] = normfield[0,:,:,:,0]*\
            (self.uumax-self.uumin)+self.uumin
            velfield[:,:,:,1] = normfield[0,:,:,:,1]*\
            (self.vvmax-self.vvmin)+self.vvmin
            velfield[:,:,:,2] = normfield[0,:,:,:,2]*\
            (self.wwmax-self.wwmin)+self.wwmin
        if self.pond == 'vel':
            velfield = np.sign(velfield)*np.sqrt(abs(velfield))
        return velfield
    

    def uvstruc_solve(self,ii,umeanfile="../../results/Simulation_3d/Umean.txt",urmsfile="../../results/Simulation_3d/Urms.txt",\
                      Hperc=1.75): 
        """
        Function for defining the Q structures in the domain
        """  
        try:
            self.UUmean 
        except:
            self.read_Umean(umeanfile)
        uu,vv,ww = self.read_velocity(ii) 
        uv = abs(np.multiply(uu,vv))
        try:
            uvi = np.multiply(self.uurms,self.vvrms) 
        except:
            self.read_Urms(urmsfile)
            uvi = np.multiply(self.uurms,self.vvrms)
        # Calculate where is the structure
        mat_struc = np.heaviside(uv-Hperc*uvi.reshape(-1,1,1),0)
        # Calculate the structure properties
        uv_str = uvstruc(mat_struc)
        uv_str.get_cluster_3D6P(uu=uu,vv=vv,flagdiv=1)
        uv_str.get_volume_cluster_box(self.y_h,self.dx,self.dz,\
                                      self.mx,self.mz,self.vol)
        uv_str.geo_char(uu,vv,self.vol,self.mx,self.my,self.mz)
        uv_str.segmentation(self.mx,self.my,self.mz)
        return uv_str
        
    def calc_uvstruc(self,start,end,umeanfile="../../results/Simulation_3d/Umean.txt",urmsfile="../../results/Simulation_3d/Urms.txt",\
                     Hperc=1.75,fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu',\
                     fold='../../results/P125_21pi_vu_Q'):
        """
        Function for calculating the uv structures
        """       
        for ii in range(start,end):
            uv_str = self.uvstruc_solve(ii)
            try:
#                from os import mkdir
                os.mkdir(fold)
            except:
                pass
            hf = h5py.File(fileQ+'.'+str(ii)+'.h5.Q', 'w')
            hf.create_dataset('Qs', data=uv_str.mat_struc)
            hf.create_dataset('Qs_event', data=uv_str.mat_event)
            hf.create_dataset('Qs_segment', data=uv_str.mat_segment)
            hf.create_dataset('dx', data=uv_str.dx)
            hf.create_dataset('dz', data=uv_str.dz)
            hf.create_dataset('ymin', data=uv_str.ymin)
            hf.create_dataset('ymax', data=uv_str.ymax)
            hf.create_dataset('vol', data=uv_str.vol)
            hf.create_dataset('volbox', data=uv_str.boxvol)
            hf.create_dataset('cdg_xbox', data=uv_str.cdg_xbox)
            hf.create_dataset('cdg_ybox', data=uv_str.cdg_ybox)
            hf.create_dataset('cdg_zbox', data=uv_str.cdg_zbox)
            hf.create_dataset('cdg_x', data=uv_str.cdg_x)
            hf.create_dataset('cdg_y', data=uv_str.cdg_y)
            hf.create_dataset('cdg_z', data=uv_str.cdg_z)
            hf.create_dataset('event', data=uv_str.event)
            hf.close()
        
    def read_uvstruc(self,ii,fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu',padpix=0):
        """
        Function for reading the uv structures
        """
        uv_str = uvstruc()
        uv_str.read_struc(ii,fileQ=fileQ,padpix=padpix)
        return uv_str
            
    def Q_perc(self,start,end,delta=1,fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu',\
               umeanfile="../../results/Simulation_3d/Umean.txt",padpix=0):
        """
        Percentage of each type
        """
        volmin = 2.7e4
        nstruc = []
        nq1 = []
        nq2 = []
        nq3 = []
        nq4 = []
        volq1 = []
        volq2 = []
        volq3 = []
        volq4 = []
        range_step = range(start,end,delta)
        for ii in range_step:
            uv_str = uvstruc()
            uv_str.read_struc(ii,fileQ=fileQ,padpix=padpix)
            lenstruc = len(uv_str.event)
            qvolmin = []
            for jj in np.arange(lenstruc):
                if uv_str.vol[jj] > volmin:
                    qvolmin.append(jj)
            uvfilter = uv_str.event[qvolmin]
            uvvolfil = uv_str.vol[qvolmin]
            lenfil = len(uvfilter)
            nstruc.append(lenfil)
            q1ind = np.where(uvfilter==1)[0]
            q2ind = np.where(uvfilter==2)[0]
            q3ind = np.where(uvfilter==3)[0]
            q4ind = np.where(uvfilter==4)[0]
            vol1 = np.sum(uvvolfil[q1ind])
            vol2 = np.sum(uvvolfil[q2ind])
            vol3 = np.sum(uvvolfil[q3ind])
            vol4 = np.sum(uvvolfil[q4ind])
            nq1.append(len(q1ind))
            nq2.append(len(q2ind))
            nq3.append(len(q3ind))
            nq4.append(len(q4ind))
            volq1.append(vol1)
            volq2.append(vol2)
            volq3.append(vol3)
            volq4.append(vol4)
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass
        volstruc = np.sum(vol1)+np.sum(vol2)+np.sum(vol3)+np.sum(vol4)
        file_save = open('../../results/Simulation_3d/bar_N_Q.txt', "w+") 
        file_save.write('Q percentage: \nOutward Int. '+\
                        str(np.sum(nq1)/lenfil)+\
                        '\nEjections '+str(np.sum(nq2)/lenfil)+\
                        '\nInward Int. '+str(np.sum(nq3)/lenfil)+\
                        '\nSweeps '+str(np.sum(nq4)/lenfil)+'\n')
        file_save.write('Q percentage total: \nOutward Int. '+\
                        str(np.sum(vol1)/np.sum(self.vol))+\
                        '\nEjections '+str(np.sum(vol2)/np.sum(self.vol))+\
                        '\nInward Int. '+str(np.sum(vol3)/np.sum(self.vol))+\
                        '\nSweeps '+str(np.sum(vol4)/np.sum(self.vol))+'\n')
        file_save.write('Q percentage structures: \nOutward Int. '+\
                        str(np.sum(vol1)/volstruc)+\
                        '\nEjections '+str(np.sum(vol2)/volstruc)+\
                        '\nInward Int. '+str(np.sum(vol3)/volstruc)+\
                        '\nSweeps '+str(np.sum(vol4)/volstruc)+'\n')
        file_save.close() 
        
    def Q_stat(self,start,end,delta=1,fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu',\
               umeanfile="../../results/Simulation_3d/Umean.txt",padpix=0):
        """
        Function for ploting Qs statistics
        """
        try:
            self.UUmean 
        except:
            try:
                self.read_Umean(umeanfile)
            except:
                self.calc_Umean(start,end)
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
        range_step = range(start,end,delta)
        for ii in range_step:
            uv_str = uvstruc()
            uv_str.read_struc(ii,fileQ=fileQ,padpix=padpix)
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
            uu,vv,ww = self.read_velocity(ii)
            uvtot = np.sum(abs(np.multiply(uu,vv)))
            uv = np.zeros((lenstruc,))
            for jj in np.arange(lenstruc):
                    indexuv = np.where(uv_str.mat_segment==jj+1)
                    for kk in np.arange(len(indexuv[0])):
                        uv[jj] += abs(uu[indexuv[0][kk],indexuv[1][kk],\
                                      indexuv[2][kk]]*vv[indexuv[0][kk],\
                                             indexuv[1][kk],indexuv[2][kk]]) 
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
        try:
            os.mkdir('../../results/Simulation_3d/')
        except:
            pass 
        cmap = cm.get_cmap('viridis', 4).colors
        fs = 20
        plt.figure()
        plt.plot(range_step,nstruc,'-',color=cmap[0,:])
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$N$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/N_struc.png')
        plt.figure()
        plt.plot(range_step,nq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(range_step,nq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(range_step,nq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(range_step,nq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$N$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/N_Q_struc.png')
        plt.figure()
        plt.plot(range_step,volq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(range_step,volq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(range_step,volq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(range_step,volq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$V^+$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/V_Q_struc.png')
        plt.figure()
        plt.plot(range_step,uvq1,'-',color=cmap[0,:],label='Q1')
        plt.plot(range_step,uvq2,'-',color=cmap[1,:],label='Q2')
        plt.plot(range_step,uvq3,'-',color=cmap[2,:],label='Q3')
        plt.plot(range_step,uvq4,'-',color=cmap[3,:],label='Q4')
        plt.xlabel('$Step$',fontsize=fs)
        plt.ylabel('$uv$',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.grid()
        plt.legend(fontsize=fs)
        plt.tight_layout()
        plt.savefig('../../results/Simulation_3d/uv_Q_struc.png')
                   
    def uvpoint(self,ii,umeanfile="../../results/Simulation_3d/Umean.txt",urmsfile="../../results/Simulation_3d/Urms.txt",\
                start=1000,end=9999):
        try:
            self.UUmean 
        except:
            try:
                self.read_Umean(umeanfile)
            except:
                self.calc_Umean(start,end)
        try:           
            self.uurms
            self.vvrms
        except:
            try:
                self.read_Urms(urmsfile)
            except:
                self.calc_rms(start,end)
        uurms = self.uurms.reshape(-1,1,1)
        vvrms = self.vvrms.reshape(-1,1,1)+1e-20
        uu,vv,ww = self.read_velocity(ii)
        index_sup = np.where(self.y_h>0)[0]
        vv[index_sup,:,:] *= -1
        uu2 = np.divide(uu,uurms)
        vv2 = np.divide(vv,vvrms)
        vec_uu = np.reshape(uu2,(-1))
        vec_vv = np.reshape(vv2,(-1))
        mean_uu = np.mean(vec_uu)
        mean_vv = np.mean(vec_vv)
        meanvec = [mean_uu,mean_vv]
        cov_mat = np.cov(np.array([vec_uu,vec_vv]))
        print(meanvec)
        print(cov_mat)
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(meanvec, cov_mat)
        import matplotlib.pyplot as plt
        xmap, ymap = np.mgrid[-3.5:3.5:.025, -3.5:3.5:0.025]
        pos = np.dstack((xmap, ymap))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(xmap,ymap,rv.pdf(pos),levels=256)
        contours=ax.contour(xmap, ymap, rv.pdf(pos),levels=8,colors='black')
        plt.clabel(contours, inline=True, fontsize=8)
        xh175_a = np.linspace(-3,-0.1,100)
        yh175_a = 1.75/xh175_a
        xh175_b = np.linspace(-3,-0.1,100)
        yh175_b = -1.75/xh175_b
        xh175_c = np.linspace(0.1,3,100)
        yh175_c = 1.75/xh175_c
        xh175_d = np.linspace(0.1,3,100)
        yh175_d = -1.75/xh175_d
        ax.plot(xh175_a,yh175_a,'--',color='black')
        ax.plot(xh175_b,yh175_b,'--',color='black')
        ax.plot(xh175_c,yh175_c,'--',color='black')
        ax.plot(xh175_d,yh175_d,'--',color='black')
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_xlabel("u/u'")
        ax.set_ylabel("v/v'")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(vec_uu,vec_vv,s=1)
        contours=ax.contour(xmap, ymap, rv.pdf(pos),levels=8,colors='black')
        plt.clabel(contours, inline=True, fontsize=8)
        xh175_a = np.linspace(-5,-0.1,100)
        yh175_a = 1.75/xh175_a
        xh175_b = np.linspace(-5,-0.1,100)
        yh175_b = -1.75/xh175_b
        xh175_c = np.linspace(0.1,5,100)
        yh175_c = 1.75/xh175_c
        xh175_d = np.linspace(0.1,5,100)
        yh175_d = -1.75/xh175_d
        ax.plot(xh175_a,yh175_a,'--',color='black')
        ax.plot(xh175_b,yh175_b,'--',color='black')
        ax.plot(xh175_c,yh175_c,'--',color='black')
        ax.plot(xh175_d,yh175_d,'--',color='black')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("u/u'")
        ax.set_ylabel("v/v'")
        
    def eval_filter(self,start,end,delta,volmin=2.7e4,\
                    fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu'):
        sumperc = 0
        range_step = range(start,end,delta)
        for ii in range_step:
            uv_str = uvstruc()
            uv_str.read_struc(ii,fileQ=fileQ)
            lenstruc = len(uv_str.vol)
            lenfilter = len(np.where(uv_str.vol<=volmin)[0])
            sumperc += lenfilter/lenstruc
        totfields = len(range_step)
        print('Percentage filtered: '+str(sumperc/totfields))
        
                    
    def eval_dz(self,start,end,delta,volmin=2.7e4,\
                    fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu'):
        deltaz_sum = 0
        lenstruc = 0
        range_step = range(start,end,delta)
        for ii in range_step:
            uv_str = uvstruc()
            uv_str.read_struc(ii,fileQ=fileQ)
            lenstruc += len(uv_str.vol>volmin)
            delta_z = uv_str.dz[uv_str.vol>volmin]*self.rey
            deltaz_sum += np.sum(delta_z)
        print('Percentage filtered: '+str(deltaz_sum/lenstruc))
        
    def eval_volfilter(self,start,end,delta,volmin=2.7e4,\
                    fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu'):
        volfilt_cum = 0
        lenstruc = 0
        range_step = range(start,end,delta)
        lenstruc = len(range_step)
        for ii in range_step:
            uv_str = uvstruc()
            uv_str.read_struc(ii,fileQ=fileQ)
            volfilt = np.sum(np.array(uv_str.vol)[np.array(uv_str.vol)<=volmin])
            volfilt_cum += np.sum(volfilt)
        mean_voltfilt = volfilt_cum/lenstruc
        print('Percentage of volume filtered: '+str(mean_voltfilt/self.voltot))
        
        
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
            os.mkdir('../../results/P125_21pi_vu_Q/')
        except:
            pass
    
    def read_struc(self,ii,fileQ='../../results/P125_21pi_vu_Q/P125_21pi_vu',padpix=0):
        fileQ_ii = fileQ+'.'+str(ii)+'.h5.Q'
        file = h5py.File(fileQ_ii, 'r')
        print('Reading: '+fileQ_ii)
        mat_struc = np.array(file['Qs'])
        mat_event = np.array(file['Qs_event'])
        mat_segment = np.array(file['Qs_segment'])
        self.dx = np.array(file['dx'])
        self.dz = np.array(file['dz'])
        self.ymin = np.array(file['ymin'])
        self.ymax = np.array(file['ymax'])
        self.vol = np.array(file['vol'])
        self.boxvol = np.array(file['volbox'])
        self.cdg_xbox = np.array(file['cdg_xbox'])
        self.cdg_ybox = np.array(file['cdg_ybox'])
        self.cdg_zbox = np.array(file['cdg_zbox'])
        self.cdg_x = np.array(file['cdg_x'])
        self.cdg_y = np.array(file['cdg_y'])
        self.cdg_z = np.array(file['cdg_z'])
        self.event = np.array(file['event'])
        if padpix > 0:
            fshape = mat_struc.shape
            dim0 = fshape[0]
            dim1 = fshape[1]
            dim2 = fshape[2]
            mat_struc_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            mat_event_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            mat_segment_pad = np.zeros((dim0,dim1+2*padpix,dim2+2*padpix))
            mat_struc_pad[:,padpix:-padpix,padpix:-padpix] = mat_struc.copy()
            mat_struc_pad[:,:padpix,padpix:-padpix] = mat_struc[:,-padpix:,:]
            mat_struc_pad[:,-padpix:,padpix:-padpix] = mat_struc[:,:padpix,:]
            mat_struc_pad[:,:,:padpix] = mat_struc_pad[:,:,-2*padpix:-padpix]
            mat_struc_pad[:,:,-padpix:] = mat_struc_pad[:,:,padpix:2*padpix]
            mat_event_pad[:,padpix:-padpix,padpix:-padpix] = mat_event.copy()
            mat_event_pad[:,:padpix,padpix:-padpix] = mat_event[:,-padpix:,:]
            mat_event_pad[:,-padpix:,padpix:-padpix] = mat_event[:,:padpix,:]
            mat_event_pad[:,:,:padpix] = mat_event_pad[:,:,-2*padpix:-padpix]
            mat_event_pad[:,:,-padpix:] = mat_event_pad[:,:,padpix:2*padpix]
            mat_segment_pad[:,padpix:-padpix,padpix:-padpix] = mat_segment.copy()
            mat_segment_pad[:,:padpix,padpix:-padpix] = mat_segment[:,-padpix:,:]
            mat_segment_pad[:,-padpix:,padpix:-padpix] = mat_segment[:,:padpix,:]
            mat_segment_pad[:,:,:padpix] = mat_segment_pad[:,:,-2*padpix:-padpix]
            mat_segment_pad[:,:,-padpix:] = mat_segment_pad[:,:,padpix:2*padpix]
            mat_struc = mat_struc_pad.copy()
            mat_event = mat_event_pad.copy()
            mat_segment = mat_segment_pad.copy()
        self.mat_struc = mat_struc
        self.mat_event = mat_event
        self.mat_segment = mat_segment
        
                        
    def get_cluster_3D6P(self,uu=[],vv=[],flagdiv=0):
        """
        Generate a sparse matrix to find the wall structures
        They are stored in a class.
        uu : velocity in u
        vv : veloicty in v
        flagdiv : divide the structures if the velocity changes sign
        """
        # Convert the structure matrix to int type, get its shape and copy
        wklocal  = self.mat_struc.astype('int')
        ny,nz,nx = wklocal.shape
        wk       = wklocal.copy()
        # Create a matrix to evaluate the connectivity-1 in all the directions
        dirs = np.array([[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1]])
        # Create vectors with the index in the directions x,z,y note that the
        # directions with the symmetry: x,z add a vector with the index of the
        # nodes, the direction y adds a -1 to indicate the presence of the wall
        indx = np.concatenate((np.arange(nx),np.arange(nx),np.arange(nx)))
        indz = np.concatenate((np.arange(nz),np.arange(nz),np.arange(nz)))
        indy = np.concatenate((np.array([-1]),np.arange(ny),np.array([-1]))) 
        # Create a vector infinitely large to store the nodes inside each 
        # structure
        pdim = 10**6      
        cola = np.zeros((3,5*pdim),dtype='int')
        nodes = np.zeros((3,pdim))
        self.nodes = []
        # Create a matrix of x,y,z for containing the index of the nodes in the
        # structures
        vrt = np.zeros((3,1),dtype='int')
        # Check the elements of the matrix
        for kk  in np.arange(0,ny):
            for jj  in np.arange(nz):
                for ii  in np.arange(nx):
                    # Skip the points where there is not a structure
                    if wk[kk,jj,ii] == 0:
                        continue
                    # The first element of cola is composed by the nodes of
                    # the element of the matrix used for calculating the 
                    # connectivity
                    cola[:,0] = np.array([kk,jj,ii],dtype='int')
                    # Initialization of the index
                    nnp = 0
                    nnq = 0
                    ssq = 0
                    while nnq <= ssq:
                        # initial point is taken from cola array, index nnq
                        # then the result is stored in nodes, index nnp
                        vrtini = cola[:,nnq]
                        if flagdiv:
                            vrtini_u = np.sign(uu[cola[0,nnq],cola[1,nnq],\
                                                  cola[2,nnq]])
                            vrtini_v = np.sign(vv[cola[0,nnq],cola[1,nnq],\
                                                  cola[2,nnq]])
                        nodes[:,nnp] = vrtini
                        # the index nnp is advanced and the studied element is
                        # removed from the matrix wk
                        nnp = nnp+1
                        wk[vrtini[0],vrtini[1],vrtini[2]] = 0
                        # All the directions in the matrix dirs are checked
                        for ld in np.arange(6):
                            vrt[0] = indy[1+dirs[ld,0]+vrtini[0]]
                            vrt[1] = indz[nz+dirs[ld,1]+vrtini[1]]
                            vrt[2] = indx[nx+dirs[ld,2]+vrtini[2]]
                            # calculate if the checked point is not a wall, 
                            # there is a structure or there was a point removed 
                            # before from the structure
                            # and check the direction of the velocity
                            while (not all(vrt[0]==-1)) and\
                            (wk[vrt[0],vrt[1],vrt[2]] == 1) and\
                            (wklocal[vrt[0],vrt[1],vrt[2]]==1) and\
                            ((not flagdiv) or\
                             (np.sign(uu[vrt[0],vrt[1],vrt[2]])==vrtini_u\
                              and np.sign(vv[vrt[0],vrt[1],vrt[2]])==\
                              vrtini_v)):
                                # Advance the index of the nodes to store
                                # Delete the stored node and repeat
                                ssq = ssq+1
                                cola[:,ssq] = vrt.reshape(3)
                                wklocal[vrt[0],vrt[1],vrt[2]] = 0
                                vrt[0] = indy[1+dirs[ld,0]+vrt[0]]
                                vrt[1] = indz[nz+dirs[ld,1]+vrt[1]]
                                vrt[2] = indx[nx+dirs[ld,2]+vrt[2]]
                        nnq += 1
                    # Define the nodes contained in the structure    
                    self.nodes.append(nodes[:,:nnp].copy()) 
                    
    def get_volume_cluster_box(self,y_h,hx,hz,mx,mz,vol):
        """
        Simplified function to calculate the approximate volume of a 3d cluster
        by calculating the volume of the containing box. The maximum and 
        minimum value are measured in every direction and multiplied  
        by the unitary distance between every point (hx,hz) and the y 
        which is not contant
        """
        # Create the information of the structures
        self.dx = np.zeros((len(self.nodes),))
        self.dz = np.zeros((len(self.nodes),))
        self.ymin = np.zeros((len(self.nodes),))
        self.ymax = np.zeros((len(self.nodes),))
        self.boxvol = np.zeros((len(self.nodes),))
        self.vol = np.zeros((len(self.nodes),))
        self.cdg_x = np.zeros((len(self.nodes),))
        self.cdg_z = np.zeros((len(self.nodes),))
        self.cdg_y = np.zeros((len(self.nodes),))
        self.cdg_xbox = np.zeros((len(self.nodes),))
        self.cdg_zbox = np.zeros((len(self.nodes),))
        self.cdg_ybox = np.zeros((len(self.nodes),))
        # Calculate for every structure
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            ymin = y_h[int(np.min(vpoints[0,:]))]
            ymax = y_h[int(np.max(vpoints[0,:]))]
            dy   = np.abs(ymax-ymin)
            x_sort = np.sort(vpoints[2,:])
            z_sort = np.sort(vpoints[1,:])
            self.cdg_xbox[nn] = np.floor(np.mean(x_sort))
            self.cdg_zbox[nn] = np.floor(np.mean(z_sort))
            self.cdg_ybox[nn] = np.floor(np.mean(vpoints[0,:])) 
            for nn2 in np.arange(len(self.nodes[nn][0,:])):
                self.cdg_x[nn] += hx*vpoints[2,nn2]*vol[vpoints[0,nn2],\
                           vpoints[1,nn2],vpoints[2,nn2]]
                self.cdg_z[nn] += hz*vpoints[1,nn2]*vol[vpoints[0,nn2],\
                           vpoints[1,nn2],vpoints[2,nn2]]
                self.cdg_y[nn] += y_h[vpoints[0,nn2]]*vol[vpoints[0,nn2],\
                           vpoints[1,nn2],vpoints[2,nn2]]
                self.vol[nn] += vol[vpoints[0,nn2],vpoints[1,nn2],vpoints[2,nn2]]
            self.cdg_x[nn] /= self.vol[nn]
            self.cdg_z[nn] /= self.vol[nn]
            self.cdg_y[nn] /= self.vol[nn]
            dx = hx*(np.max(x_sort)-np.min(x_sort))
            dz = hz*(np.max(z_sort)-np.min(z_sort)) 
            # Check if the structure is crossing the symmetry planes x and z
            flag_x = np.count_nonzero(x_sort==mx-1)>= 1 and\
            np.count_nonzero(x_sort==0)>=1
            flag_z = np.count_nonzero(z_sort==mz-1)>= 1 and\
            np.count_nonzero(z_sort==0)>=1    
            # If the structure is crossing the x plane
            if flag_x:
                # take the unique indexex of the x position and mark those that
                # are not in contact with the structure matching the x plane
                x_uni = np.unique(x_sort)
                aux = x_uni == np.arange(len(x_uni))
                imin = np.where(aux==0)[0]  
                # If the structure is crossing the symmetry plane and divided
                if not len(imin) == 0:
                    # calculate the minimum and maximum indexes in the 
                    # separated part of the structure. The minimum value is the
                    # minimum of the last structure, the maximum is the 
                    # maximum of the first structure
                    xmin = x_uni[imin[0]]
                    xmax = mx+x_uni[imin[0]-1]
                    dxsimp = xmax-xmin
                    dx = hx*dxsimp.astype('double')
                    tmp = x_sort
                    II = tmp<=x_uni[imin[0]-1]
                    tmp[II] = tmp[II] + mx
                    self.cdg_xbox[nn] = np.mod(np.floor(np.mean(tmp)),mx)
                    tmp2 = vpoints[2,:].copy()
                    JJ = tmp2<=x_uni[imin[0]-1]
                    tmp2[JJ] = tmp2[JJ]+mx
                    self.cdg_x[nn] = 0
                    for nn2 in np.arange(len(self.nodes[nn][0,:])):
                        self.cdg_x[nn] += hx*tmp2[nn2]*\
                        vol[vpoints[0,nn2],vpoints[1,nn2],vpoints[2,nn2]]
                    self.cdg_x[nn] /= self.vol[nn]
                    if self.cdg_x[nn] > hx*mx:
                        self.cdg_x[nn] -= hx*mx
            # If the structure is crossing the z plane
            if flag_z:
                # take the unique indexex of the x position and mark those that
                # are not in contact with the structure matching the z plane
                z_uni = np.unique(z_sort)
                aux = z_uni == np.arange(len(z_uni))
                imin = np.where(aux==0)[0]  
                # If the structure is crossing the symmetry plane and divided             
                if not len(imin) == 0:
                    # calculate the minimum and maximum indexes in the 
                    # separated part of the structure. The minimum value is the
                    # minimum of the last structure, the maximum is the 
                    # maximum of the first structure
                    zmin = z_uni[imin[0]]
                    zmax = mz+z_uni[imin[0]-1]
                    dzsimp = zmax-zmin
                    dz = hz*dzsimp.astype('double')
                    tmp = z_sort
                    II = tmp<=z_uni[imin[0]-1]
                    tmp[II] = tmp[II]+mz
                    self.cdg_zbox[nn] = np.mod(np.floor(np.mean(tmp)),mz)
                    tmp2 = vpoints[1,:].copy()
                    JJ = tmp2<=z_uni[imin[0]-1]
                    tmp2[JJ] = tmp2[JJ]+mz
                    self.cdg_z[nn] = 0
                    for nn2 in np.arange(len(self.nodes[nn][0,:])):
                        self.cdg_z[nn] += hz*tmp2[nn2]*\
                        vol[vpoints[0,nn2],vpoints[1,nn2],vpoints[2,nn2]]
                    self.cdg_z[nn] /= self.vol[nn]
                    if self.cdg_z[nn] > hz*mz:
                        self.cdg_z[nn] -= hz*mz
            self.dx[nn] = dx
            self.dz[nn] = dz
            self.ymin[nn] = ymin
            self.ymax[nn] = ymax
            self.boxvol[nn] = dy*dx*dz
                    
                    
    def geo_char(self,du,dv,vol,mx,my,mz):
        """
        Function for calculating the geometrical characteristics of the uv 
        structures
        """
        # define the type of event matrix and the volume of each event
        self.mat_event = np.zeros((my,mz,mx))
        self.event = np.zeros((len(self.nodes),))
        # Evaluate the characteristics for each structure
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            voltot = np.zeros((4,))
            # Evaluate each node of the structure
            for nn_node in np.arange(len(vpoints[0,:])):
                # get the u and v velocities for each point of the structure
                duval = du[vpoints[0,nn_node],vpoints[1,nn_node],\
                        vpoints[2,nn_node]]
                if self.cdg_y[nn] <= 0:
                    dvval = dv[vpoints[0,nn_node],\
                               vpoints[1,nn_node],vpoints[2,nn_node]]
                else:
                    dvval = -dv[vpoints[0,nn_node],\
                                vpoints[1,nn_node],vpoints[2,nn_node]]
                vol_nod = np.sqrt(duval**2+dvval**2)*\
                vol[vpoints[0,nn_node],vpoints[1,nn_node],\
                    vpoints[2,nn_node]]
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
                self.mat_event[vpoints[0,nn_node],vpoints[1,nn_node],\
                               vpoints[2,nn_node]] = self.event[nn]

                
                
    def segmentation(self,mx,my,mz):
        """
        Function to segment the model
        """
        self.mat_segment = np.zeros((my,mz,mx))
        for nn  in np.arange(len(self.nodes)):
            vpoints = self.nodes[nn].astype('int')
            for nn_node in np.arange(len(vpoints[0,:])):
                self.mat_segment[vpoints[0,nn_node],vpoints[1,nn_node],\
                               vpoints[2,nn_node]] = nn+1
                
        
        