import re,operator
from multiprocessing import Pool
import numpy as np
from os import listdir
import scipy.stats as scs
import time as time
import sklearn.linear_model as lm
import sklearn.ensemble as ens
import re
import scipy.signal as ss
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn import (decomposition,lda)
from copy import deepcopy
#import pandas as pd
np.set_printoptions(precision=10)
class SignalTools():
  def movingAve(self,v,s=2):
      ret=np.cumsum(v,dtype=float,axis=0)
      ret[s:,:]=ret[s:,:]-ret[:-s,:]
      return  ret[s-1:,:]/s
    
  def medFilter(self,my_data,kernel_size=(3,1)):
      return ss.medfilt(my_data,kernel_size=kernel_size)
  
  def fft(self,my_data,axis=-1):
      return np.fft.fft(my_data,axis=axis)
  def psd(self,my_data,top=5):
      N=my_data.shape[0]
      freq_ind=np.arange(1,N/2+1)
      my_fft=self.fft(my_data)
      my_psd=abs(my_fft[freq_ind])**2+abs(my_fft[-freq_ind])**2
      h_ind=(-my_psd).argsort()[:top]
      #h_ind,=np.where(my_psd>threshold)
      freq=np.fft.fftfreq(N)
      #print(h_ind.size,my_psd.shape,np.where(my_psd>threshold)[0])
      #if h_ind.size>0:
      #print (my_psd)
      return freq[freq_ind[h_ind]],my_psd[h_ind]
      #else:
      #    return np.zeros((1,2))
  def convolution(self,my_data,conv,mode='valid'):
      return np.convolve(my_data,conv,mode=mode)
  
class FeatureTools(SignalTools):
  
  def pcaFit(self,X,n_components=20,verbose=False):
      reduction_method=decomposition.PCA(n_components=n_components,whiten=True)
      reduction_method.fit(X)
      if verbose:
        print ("explained variance ration: " ,reduction_method.explained_variance_ratio_)
        print("sum of explained variance ratio %.2f for %i components"%(np.sum(reduction_method.explained_variance_ratio_),n_components))
      return reduction_method.transform(X)
	      
  def vecCrossAngle(self,v1,v2):
      temp=(v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0])/(self.vecAbs(v1)*self.vecAbs(v2))
      temp=temp[~np.isnan(temp)]
      temp=temp[~np.isinf(temp)]
      temp[np.abs(temp)>1.0]=np.sign(temp[np.abs(temp)>1.0])  
      return  np.arcsin(temp) 
  
  def vecCross(self,v1,v2):
      return  v1[:,0]*v2[:,1]-v1[:,1]*v2[:,0]
  
  def vecDot(self,v1,v2):
      return v1[:,0]*v2[:,0]+ v1[:,1]*v2[:,1] 
    
  def vecAngle(self,v1,v2):
      temp=(v1[:,0]*v2[:,0]+ v1[:,1]*v2[:,1])/(self.vecAbs(v1)*self.vecAbs(v2))
      temp=temp[~np.isnan(temp)]
      temp=temp[~np.isinf(temp)]
      temp[np.abs(temp)>1.0]=np.sign(temp[np.abs(temp)>1.0])
      return np.arccos(temp)
  def vecWierdAngle(self,v1,v2):
      return np.arccos((v1[:,0]*v2[:,0]+ v1[:,1]*v2[:,1])/(self.vecAbs(v1)+self.vecAbs(v2)))

  def vecAbs(self,vec):
      return np.sqrt(vec[:,0]**2+vec[:,1]**2)

  def velVec(self,my_data,n=1):
      return np.diff(my_data,n=n,axis=0) 

  def findMinMax(self,my_data):
      return [min(my_data),max(my_data)]
  
  def accVec(self,my_data,n=1):
      vel_vec_temp=self.velVec(my_data,n=n)
      return self.velVec(vel_vec_temp,n=1)
  
  def hist(self,my_data,bins=50,density=True,range=None):
      if bins==None and range==None: bins=np.arange(min(my_data),max(my_data),(max(my_data)-min(my_data))/10)
      return np.histogram(my_data,bins=bins,range=range,density=density)
      #values,base=np.histogram(my_data,bins=bins,range=range)
      #if density:
	#try:
	 # values/np.sum(values)
	#except:
	 # print("except: ",np.sum(values),np.max(values),np.min(values))
	#return values/np.sum(values),base
      #else:
        #return values,base
  

  def velQuant(self,my_data,prob=np.arange(0,1.05,0.05),n=1):
      vel_vec_temp=self.velVec(my_data,n=n)
      vel_vec_abs=self.vecAbs(vel_vec_temp)
      return scs.mstats.mquantiles(vel_vec_abs,prob=prob,alphap=1,betap=1,axis=0)

  def accQuant(self,my_data,prob=np.arange(0,1.05,0.05),n=1):
      acc_vec_temp=self.accVec(my_data,n=n)
      acc_vec_abs=self.accAbs(acc_vec_temp)
      return scs.mstats.mquantiles(acc_vec_abs,prob=prob,aphap=1,betap=1,axis=0)

  
  
  def makeFeatures(self,my_data,vel_max=90,acc_max=45,n=1):
      ##data cleaning and smoothing 
      #my_data=self.medFilter(my_data,kernel_size=(3,1))
      #my_data=self.movingAve(my_data,s=3)
      ##making vel feature 
      my_vel=self.velVec(my_data,n=n)
      my_vel_abs=self.vecAbs(my_vel)
      vel_feature,edge=self.hist(my_vel_abs,range=(0,40),bins=50)
      my_total_trip_length=np.zeros((1))
      my_total_trip_length[0]=np.sum(my_vel_abs)
      ##making acc feature
      my_acc=self.velVec(my_vel,n=n)
      my_acc_vel_dot=self.vecDot(my_vel[0:-1,:],my_acc)
      my_acc_ws=self.vecAbs(my_acc)*np.sign(my_acc_vel_dot)
      my_acc_ws=my_acc_ws[(my_acc_ws<18) & (my_acc_ws>-36)]
      acc_feature,edge=self.hist(my_acc_ws,range=(-36,18),bins=120)
      
      ##making accdiff feature
      my_acc_diff=self.velVec(my_acc,n=n)
      my_acc_diff_abs=self.vecAbs(my_acc_diff)
      acc_diff_feature,edge=self.hist(my_acc_diff_abs,range=(0,5),bins=15)
      
      
      ##making conv based feature for acc
      ##immediate start stop pattern
      conv=[1,-1]
      imm_start_stop_pattern=self.convolution(my_acc_ws,conv)
      imm_start_stop_pattern_hist,pattern_edge=self.hist(imm_start_stop_pattern,bins=10)
      imm_start_stop_pattern_hist_hidx=(imm_start_stop_pattern_hist).argsort()
      imm_start_stop_pattern_hist_feature=pattern_edge[imm_start_stop_pattern_hist_hidx[-2:]]
      
      ##making vel acc angle feature
      #epsilon=0.000001
      #bins0_incr=np.pi/90
      #bins0=np.arange(0,np.pi/6-epsilon,bins0_incr)
      #bins1_incr=np.pi/36
      #bins1=np.arange(np.pi/6,np.pi*5/6-epsilon,bins1_incr)
      #bins2=np.arange(np.pi*5/6,np.pi+bins0_incr,bins0_incr)
      #acc_vel_angle_bins=np.concatenate((bins0,bins1,bins2))
      #print("vel_acc",180.0/np.pi*acc_vel_angle_bins )
      #my_acc_vel_angle=self.vecAngle(my_vel[0:-1,:],my_acc)
      #print("my_acc_vel_angle nan",my_acc_vel_angle[~np.isfinite(my_acc_vel_angle)])
      #acc_vel_angle_feature,edge=self.hist(my_acc_vel_angle)#,bins=acc_vel_angle_bins)
      #,range=(0,np.pi))
      ###wierd feature
      #my_acc_vel_wierd_angle=self.vecWierdAngle(my_vel[0:-1,:],my_acc)
      #acc_vel_wierd_angle_feature,edge=self.hist(my_acc_vel_wierd_angle,bins=50,range=(0,np.pi))

      
      
      
      my_vel_vel_cross=self.vecCross(my_vel[0:-2,:],my_vel[1:-1,:])
      ###### angular velocity 
      my_vel_acc_cross=self.vecCross(my_vel[0:-1,:],my_acc)
      my_vel_acc_cross_abs=np.abs(my_vel_acc_cross)
      my_ang_vel=my_vel_acc_cross_abs/np.sqrt(my_vel_abs[0:-1])
      #my_ang_vel[~np.isfinite(my_ang_vel)]=0.0
      #my_ang_vel=my_ang_vel[np.isfinite(my_ang_vel)]
      #try:
     
      
      ang_vel_feature,edge=self.hist( my_ang_vel,range=(0,15),bins=15)
      
      my_vel_acc_cross_feature,edge=self.hist(my_vel_acc_cross,range=(-40,40),bins=80)
      #except(RuntimeWarning):
	#print("except: ",np.sum(my_ang_vel),np.max(my_ang_vel),np.min(my_ang_vel))
      ##making vel acc dot feature
      #my_acc_vel_dot=self.vecDot(my_vel[0:-1],my_acc)
      #acc_vel_dot_feature,edge=self.hist(my_acc_vel_dot,range=(-500,500))
      
      #making vel vel cross angle feature
      
      #bins0_incr=np.pi/36
      #bins0=np.arange(-np.pi/2,-np.pi/6-epsilon,bins0_incr)
      #bins1_incr=np.pi/90
      #bins1=np.arange(-np.pi/6,np.pi/6-epsilon,bins1_incr)
      #bins2=np.arange(np.pi/6,np.pi/2+bins0_incr,bins0_incr)
      #vel_vel_cross_angle_bins=np.concatenate((bins0,bins1,bins2))
      #print("vel_vel",np.diff(vel_vel_cross_angle_bins) ,180.0/np.pi*vel_vel_cross_angle_bins)
      #my_vel_vel_cross_angle=self.vecCrossAngle(my_vel[0:-2,:],my_vel[1:-1,:])
      #print("my_vel_vel_cross_angle nan",my_vel_vel_cross_angle[~np.isfinite(my_vel_vel_cross_angle)])
      #vel_vel_cross_angle_feature,edge=self.hist(my_vel_vel_cross_angle)#,bins=vel_vel_cross_angle_bins)
      #range=(-np.pi/2.0,np.pi/2.0)) 
      
      ## frequency_based features
      my_vel_freq_feature,temp=self.psd(my_vel_abs)
      #my_acc_pos=deepcopy(my_acc_ws)
      #my_acc_neg=deepcopy(my_acc_ws)
      #my_acc_pos[np.sign(my_acc_ws)<0]=0
      #my_acc_neg[np.sign(my_acc_ws)>0]=0
      #my_acc_pos_freq_feature,temp=self.psd( my_acc_pos)
      #my_acc_neg_freq_feature,temp=self.psd( my_acc_neg)
      my_acc_freq_feature,temp=self.psd( my_acc_ws)
      my_acc_vel_dot_freq_feature,temp=self.psd(my_acc_vel_dot)
      my_vel_vel_cross_freq_feature,temp=self.psd(my_vel_vel_cross)
      my_ang_vel_freq_feature,temp=self.psd(my_ang_vel)
      #my_acc_vel_angle_freq_feature,temp=self.psd(my_acc_vel_angle)
      #if my_acc_vel_angle_freq_feature.shape[0] !=5:  print("my_acc_vel_angle_freq_feature.shape: ", my_acc_vel_angle_freq_feature.shape,my_acc_vel_angle_freq_feature )
      
      return np.concatenate((my_total_trip_length,vel_feature,acc_feature,acc_diff_feature,ang_vel_feature,my_vel_acc_cross_feature,imm_start_stop_pattern_hist_feature,
          my_vel_freq_feature,my_acc_vel_dot_freq_feature,my_vel_vel_cross_freq_feature,my_ang_vel_freq_feature))
          #acc_vel_angle_feature,vel_vel_cross_angle_feature)) 
            #acc_vel_dot_feature))#acc_vel_angle_feature,acc_diff_feature))
  
class ModelingTools():
  def __init__(self,all_data,all_identifiers,all_weights=None,samples_per_driver=200,ensemble_size=1,ignore_sample_size=None,verbose=0):
      self.verbose=verbose
      self.ensemble_size=ensemble_size
      self.all_data=all_data
      self.all_weights=all_weights
      self.all_identifiers=all_identifiers
      self.all_range=set(range(all_data.shape[0]))
      self.number_of_features=self.all_data.shape[1]
      self.samples_per_driver=samples_per_driver
      self.number_of_drivers=(self.all_data.shape[0])/self.samples_per_driver
      self.y=np.zeros(3*self.samples_per_driver)
      self.X=np.zeros((3*self.samples_per_driver,self.number_of_features))
      self.y[0:self.samples_per_driver]=1.0
      self.weights=np.ones((3*self.samples_per_driver))
      self.ignore_sample_size=ignore_sample_size
      if self.ignore_sample_size!=None:
	self.fit_sample_size=self.number_of_drivers-self.ignore_sample_size
	self.X_pred=np.zeros((self.samples_per_driver,self.number_of_features))
      if self.verbose: 
          print("Number of Features: ",self.number_of_features)
          print("Number of Drivers: ",self.number_of_drivers)
          if self.all_weights!=None: print("Number of weights: ",self.all_weights.shape)
  def logisticReg(self,X,y,**kwargs):
      #for key,value in kwargs.iteritems():
	    #setattr(self,key,value)
      logr=lm.LogisticRegression(**kwargs)
      logr.fit(X,y)
      return logr.predict_proba(X)[:,1]
  
  def findRangeOfRows(self,driver):
      if self.verbose>2: print("driver ",driver," range: ",
              driver*self.samples_per_driver,(driver+1)*self.samples_per_driver )
      return driver*self.samples_per_driver,(driver+1)*self.samples_per_driver

  def doLogRegEnsemble(self,driver,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      ### first and last+1 rows correspond to "driver"
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X[0:self.samples_per_driver,:]=self.all_data[my_driver_f_row:my_driver_l_row,:]
      ensemble_of_predictions=[]
      for i in range(self.ensemble_size):
        other_driver=np.random.randint(0,self.number_of_drivers-1)      
        ##make sure we are not choosing my_driver as the other_driver too!! 
        while  other_driver==driver:
          print('bad luck!!')
          other_driver=np.random.randint(0,self.number_of_drivers-1)
        other_driver_f_row,other_driver_l_row=self.findRangeOfRows(other_driver)
        self.X[self.samples_per_driver:2*self.samples_per_driver,:]= \
            self.all_data[other_driver_f_row:other_driver_l_row,:]
        ensemble_of_predictions.append(
                self.logisticReg(self.X,self.y,**kwargs)[0:self.samples_per_driver])
      
      ensemble_of_predictions=np.vstack(tuple(ensemble_of_predictions))
      if self.verbose>1: 
          print("Ensemble of Predictions` shape: ",
              ensemble_of_predictions.shape)
          print("Ensemble of Predictions` head: ",ensemble_of_predictions[:,0:self.verbose])

      my_driver_final_prediction=np.mean(ensemble_of_predictions,axis=0)
      
      if self.verbose>1: 
          print("Shape of Final predictions for the driver ", driver," ",
              my_driver_final_prediction.shape )
          print("Final predictions` head: ",my_driver_final_prediction[0:self.verbose])
      
      return my_driver_final_prediction, \
              self.all_identifiers[my_driver_f_row:my_driver_l_row]
  
  def gradientBoosting(self,X,y,X_pred=None,sample_weight=None,**kwargs):
      #for key,value in kwargs.iteritems():
	    #setattr(self,key,value)
      clf=ens.GradientBoostingClassifier(**kwargs)
      clf.fit(X,y)
      if self.verbose>0 and X_pred!=None: return clf.predict_proba(X_pred)[:,1],clf.feature_importances_
      if X_pred!=None: return clf.predict_proba(X_pred)[:,1]
      else: return clf.predict_proba(X)[:,1]
  
  def randomForest(self,X,y,X_pred=None,sample_weight=None,**kwargs):
      #for key,value in kwargs.iteritems():
	    #setattr(self,key,value)
      clf=ens.RandomForestClassifier(**kwargs)
      clf.fit(X,y,sample_weight=sample_weight)
      if self.verbose>0 and X_pred!=None: return clf.predict_proba(X_pred)[:,1],clf.feature_importances_
      if X_pred!=None: return clf.predict_proba(X_pred)[:,1]
      else: return clf.predict_proba(X)[:,1]

  
  def doRFEnsemble(self,driver,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      ### first and last+1 rows correspond to "driver"
      
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X[0:self.samples_per_driver,:]=self.all_data[my_driver_f_row:my_driver_l_row,:]
      ensemble_of_predictions=[]
      #other_driver_range=list(self.all_range-set(range(my_driver_f_row,my_driver_l_row)))
      for i in range(self.ensemble_size):
        #other_driver=np.random.randint(0,self.number_of_drivers-1)      
        ##make sure we are not choosing my_driver as the other_driver too!! 
        #while  other_driver==driver:
          #print('bad luck!!')
          #other_driver=np.random.randint(0,self.number_of_drivers-1)
        #other_driver_f_row,other_driver_l_row=self.findRangeOfRows(other_driver)
        #other_driver_idx=np.random.choice(other_driver_range,self.samples_per_driver,replace=False)
        other_driver_idx=other_drivers_random_sampled[500*driver+i,:]
        ###correct for the fact that we don't want to choose other drivers from my driver
        other_driver_idx[other_driver_idx>=my_driver_f_row]+=self.samples_per_driver
        self.X[self.samples_per_driver:,:]= \
            self.all_data[other_driver_idx,:]
	  
        if self.all_weights!=None:self.weights[:self.samples_per_driver]=self.all_weights[my_driver_f_row:my_driver_l_row]
        ensemble_of_predictions.append(
                self.randomForest(self.X,self.y,sample_weight=self.weights,**kwargs)[0:self.samples_per_driver])
      
      ensemble_of_predictions=np.vstack(tuple(ensemble_of_predictions))
      if self.verbose>1: 
          print("Ensemble of Predictions` shape: ",
              ensemble_of_predictions.shape)
          print("Ensemble of Predictions` head: ",ensemble_of_predictions[:,0:self.verbose])

      my_driver_final_prediction=np.mean(ensemble_of_predictions,axis=0)
      
      if self.verbose>1: 
          print("Shape of Final predictions for the driver ", driver," ",
              my_driver_final_prediction.shape )
          print("Final predictions` head: ",my_driver_final_prediction[0:self.verbose])
      
      return my_driver_final_prediction, \
              self.all_identifiers[my_driver_f_row:my_driver_l_row]
	    
  def doRFEnsembleIter(self,driver,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      #ignore_sample_size=20
      #fit_sample_size=self.number_of_drivers-ignore_sample_size
      #y_ignore=np.zeros((2*fit_sample_size))
      #y_ignore[0:fit_sample_size]=1.0
      ### first and last+1 rows correspond to "driver"
      
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X_pred=self.all_data[my_driver_f_row:my_driver_l_row,:]
      self.weights=self.all_weights[my_driver_f_row:my_driver_l_row]
      if self.verbose>1:print("all_weights",self.weights[0:self.verbose],(self.all_identifiers[my_driver_f_row:my_driver_l_row])[0:self.verbose])
      ordered_weights_idx=self.weights.argsort()
      if self.verbose>1: 
        print("ordered_weights_idx",ordered_weights_idx.shape,ordered_weights_idx[0:self.verbose],
                  (self.all_identifiers[my_driver_f_row:my_driver_l_row])[ordered_weights_idx[0:self.verbose]])
      #print("ordered_weights_idx",ordered_weights_idx.shape,ordered_weights_idx[0:10],
      #            (self.all_identifiers[my_driver_f_row:my_driver_l_row])[ordered_weights_idx[0:10]])
      ### sort X ascendingly (bases on weights)
      self.X[0:self.samples_per_driver,:]=self.X_pred[ordered_weights_idx,:]
      ensemble_of_predictions=[]
      other_driver_range=list(self.all_range-set(range(my_driver_f_row,my_driver_l_row)))
      
      for i in range(self.ensemble_size):
        #other_driver=np.random.randint(0,self.number_of_drivers-1)      
        ##make sure we are not choosing my_driver as the other_driver too!! 
        #while  other_driver==driver:
          #print('bad luck!!')
          #other_driver=np.random.randint(0,self.number_of_drivers-1)
        #other_driver_f_row,other_driver_l_row=self.findRangeOfRows(other_driver)
        other_driver_idx=np.random.choice(other_driver_range,self.samples_per_driver,replace=False)
        #self.X[self.samples_per_driver:,:]= \
        #    self.all_data[other_driver_f_row:other_driver_l_row,:]
	self.X[self.samples_per_driver:,:]= \
            self.all_data[other_driver_idx,:]

        #ensemble_of_predictions.append(
        #        self.randomForest(self.X[self.ignore_sample_size:-self.ignore_sample_size,:],self.y[self.ignore_sample_size:-#self.ignore_sample_size],self.X_pred,**kwargs)[0:self.samples_per_driver])
        
        ensemble_of_predictions.append(
                self.randomForest(self.X[self.ignore_sample_size:-self.ignore_sample_size,:],self.y[self.ignore_sample_size:-self.ignore_sample_size],self.X_pred,**kwargs)[0:self.samples_per_driver]) 
      
      ensemble_of_predictions=np.vstack(tuple(ensemble_of_predictions))
      if self.verbose>1: 
          print("Ensemble of Predictions` shape: ",
              ensemble_of_predictions.shape)
          print("Ensemble of Predictions` head: ",ensemble_of_predictions[:,0:self.verbose])

      my_driver_final_prediction=np.mean(ensemble_of_predictions,axis=0)
      
      if self.verbose>1: 
          print("Shape of Final predictions for the driver ", driver," ",
              my_driver_final_prediction.shape )
          print("Final predictions` head: ",my_driver_final_prediction[0:self.verbose])
      
      return my_driver_final_prediction, \
              self.all_identifiers[my_driver_f_row:my_driver_l_row]	    

  def doGBEnsemble(self,driver,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      ### first and last+1 rows correspond to "driver"
      
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X[0:self.samples_per_driver,:]=self.all_data[my_driver_f_row:my_driver_l_row,:]
      ensemble_of_predictions=[]
      #other_driver_range=list(self.all_range-set(range(my_driver_f_row,my_driver_l_row)))
      for i in range(self.ensemble_size):
        #other_driver=np.random.randint(0,self.number_of_drivers-1)      
        ##make sure we are not choosing my_driver as the other_driver too!! 
        #while  other_driver==driver:
          #print('bad luck!!')
          #other_driver=np.random.randint(0,self.number_of_drivers-1)
        #other_driver_f_row,other_driver_l_row=self.findRangeOfRows(other_driver)
        #other_driver_idx=np.random.choice(other_driver_range,self.samples_per_driver,replace=False)
        other_driver_idx=other_drivers_random_sampled[500*driver+i,:]
        ###correct for the fact that we don't want to choose other drivers from my driver
        other_driver_idx[other_driver_idx>=my_driver_f_row]+=self.samples_per_driver
        self.X[self.samples_per_driver:,:]= \
            self.all_data[other_driver_idx,:]
	  
        if self.all_weights!=None:self.weights[:self.samples_per_driver]=self.all_weights[my_driver_f_row:my_driver_l_row]
        ensemble_of_predictions.append(
                self.gradientBoosting(self.X,self.y,X_pred=self.X[:self.samples_per_driver],**kwargs)[0:self.samples_per_driver])
      
      ensemble_of_predictions=np.vstack(tuple(ensemble_of_predictions))
      if self.verbose>1: 
          print("Ensemble of Predictions` shape: ",
              ensemble_of_predictions.shape)
          print("Ensemble of Predictions` head: ",ensemble_of_predictions[:,0:self.verbose])

      my_driver_final_prediction=np.mean(ensemble_of_predictions,axis=0)
      
      if self.verbose>1: 
          print("Shape of Final predictions for the driver ", driver," ",
              my_driver_final_prediction.shape )
          print("Final predictions` head: ",my_driver_final_prediction[0:self.verbose])
      
      return my_driver_final_prediction, \
              self.all_identifiers[my_driver_f_row:my_driver_l_row]
	    


  def adaBoost(self,X,y,**kwargs):
      adb=ens.AdaBoostClassifier(**kwargs)
      adb.fit(X,y)
      if self.verbose>0: return adb.predict_proba(X)[:,1],adb.feature_importances_
      else: return adb.predict_proba(X)[:,1]
  
  def gradBoost(self,X,y,**kwargs):
      gb=ens.GradientBoostingClassifier(**kwargs)
      gb.fit(X,y)
      if self.verbose>0: return gb.predict_proba(X)[:,1],gb.feature_importances_
      else: return gb.predict_proba(X)[:,1]
  
  def doADBEnsemble(self,driver,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      ### first and last+1 rows correspond to "driver"
      
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X[0:self.samples_per_driver,:]=self.all_data[my_driver_f_row:my_driver_l_row,:]
      ensemble_of_predictions=[]
      for i in range(self.ensemble_size):
        other_driver=np.random.randint(0,self.number_of_drivers-1)      
        ##make sure we are not choosing my_driver as the other_driver too!! 
        while  other_driver==driver:
          print('bad luck!!')
          other_driver=np.random.randint(0,self.number_of_drivers-1)
        other_driver_f_row,other_driver_l_row=self.findRangeOfRows(other_driver)
        self.X[self.samples_per_driver:2*self.samples_per_driver,:]= \
            self.all_data[other_driver_f_row:other_driver_l_row,:]
        ensemble_of_predictions.append(
                self.adaBoost(self.X,self.y,**kwargs)[0:self.samples_per_driver])
      
      ensemble_of_predictions=np.vstack(tuple(ensemble_of_predictions))
      if self.verbose>1: 
          print("Ensemble of Predictions` shape: ",
              ensemble_of_predictions.shape)
          print("Ensemble of Predictions` head: ",ensemble_of_predictions[:,0:self.verbose])

      my_driver_final_prediction=np.mean(ensemble_of_predictions,axis=0)
      
      if self.verbose>1: 
          print("Shape of Final predictions for the driver ", driver," ",
              my_driver_final_prediction.shape )
          print("Final predictions` head: ",my_driver_final_prediction[0:self.verbose])
      
      return my_driver_final_prediction, \
              self.all_identifiers[my_driver_f_row:my_driver_l_row]
  


  def reduceResult(self,results):
      all_data=[driver[0] for driver in results]
      all_identifiers=[driver[1]for driver in results]
      all_data=np.hstack(tuple(all_data))
      all_identifiers=np.hstack(tuple(all_identifiers))
      return all_data,all_identifiers

  def randomDriverProducer(self,out_file="other_drivers_sample.npz"):
      other_driver_range=((self.all_data.shape)[0])-self.samples_per_driver
      total_range_list=range(other_driver_range)
      ensamble_of_samples=[]
      for i in range(500*self.number_of_drivers):
          ensamble_of_samples.append(np.random.choice(total_range_list,self.samples_per_driver,replace=False))
      
      ensamble_of_samples=np.vstack(tuple(ensamble_of_samples))
      np.savez(out_file,ensamble_of_samples)




  


class SemiSupervisedLearningTools(ModelingTools):
  def __init__(self,all_priors,all_prior_identifiers,all_data,all_identifiers,samples_per_driver=200
          ,verbose=0,ensemble_size=100):
      self.all_priors=np.array(all_priors)
      self.all_prior_identifiers=np.array(all_prior_identifiers)
      self.verbose=verbose
      self.all_data=all_data
      self.all_identifiers=all_identifiers
      self.number_of_features=self.all_data.shape[1]
      self.samples_per_driver=samples_per_driver
      self.number_of_drivers=(self.all_data.shape[0])/self.samples_per_driver
      self.y=np.zeros(self.samples_per_driver)
      self.X=np.zeros((self.samples_per_driver,self.number_of_features))
      self.y[0:self.samples_per_driver]=-1.0
      if self.verbose: 
          print("Number of Features: ",self.number_of_features)
          print("Number of Drivers: ",self.number_of_drivers)
      self.ensemble_size=ensemble_size
  
  
  def doLabelPropagation(self,X,y,**kwargs):
      label_prop_model = LabelPropagation(**kwargs)
      if self.verbose>2: 
          print("X, y shapes: ",X.shape,y.shape)
          print(" y hist: ",np.histogram(y))
      label_prop_model.fit(X, y)
      if self.verbose>2: print("lp_predict:",np.histogram(label_prop_model.predict(X)) )
      return label_prop_model.predict_proba(X)
  
  def doLabelSpreading(self,X,y,**kwargs):
      label_spread_model = LabelSpreading(**kwargs)
      if self.verbose>2: 
          print("X, y shapes: ",X.shape,y.shape)
          print(" y hist: ",np.histogram(y))
      label_spread_model.fit(X, y)
      if self.verbose>2: print("ls_predict:",np.histogram(label_spread_model.predict(X)) )
      return label_spread_model.predict_proba(X)


  def myDriverPriors(self,first_identifier):
      my_driver_first_row_in_priors=np.where(self.all_prior_identifiers==first_identifier)[0][0]
      if self.verbose: print("my_driver_first_row_in_priors: ",my_driver_first_row_in_priors)
      my_driver_last_row_in_priors=my_driver_first_row_in_priors+self.samples_per_driver
      return self.all_priors[my_driver_first_row_in_priors:my_driver_last_row_in_priors]
  
  def findNHighestLowestidx(self,my_array,n=5):
      my_array_sorted=np.sort(my_array)
      n_highest_idx=np.where(my_array>my_array_sorted[-(50+1)])[0]
      n_lowest_idx=np.where(my_array<my_array_sorted[n])[0]
      if self.verbose>2: print("highest and lowest idx: ",
              n_highest_idx,n_lowest_idx)
      return n_highest_idx,n_lowest_idx

  def bayesianEstimate(self,driver,ssa='ls',number_of_known_samples=50,**kwargs):
      if self.verbose: print("Driver Number: ",driver)
      ### first and last+1 rows correspond to "driver"
      
      my_driver_f_row,my_driver_l_row=self.findRangeOfRows(driver)
      self.X=self.all_data[my_driver_f_row:my_driver_l_row,:]
      
      my_dirver_priors=self.myDriverPriors(self.all_identifiers[my_driver_f_row])
      my_dirver_priors_min=np.min(my_dirver_priors)
      my_dirver_priors_max=np.max(my_dirver_priors)
      if self.verbose>2: print("my_dirver_priors_min",my_dirver_priors_min);print("my_dirver_priors_max",my_dirver_priors_max)
      my_dirver_scaled_prior=(my_dirver_priors-my_dirver_priors_min)/(my_dirver_priors_max-my_dirver_priors_min)
      #my_dirver_scaled_prob=my_dirver_scaled_prior/np.sum(my_dirver_scaled_prior)
      #other_dirver_scaled_prior=1.000-my_dirver_scaled_prior
      #other_dirver_scaled_prob=other_dirver_scaled_prior/np.sum(other_dirver_scaled_prior)
      #if self.verbose>1:print("my_dirver_scaled_prob:", my_dirver_scaled_prob);print("other_dirver_scaled_prob:", other_dirver_scaled_prob)
      #if self.all_identifiers[my_driver_f_row]=='2442_1': print("my_driver_prior_hist:",
       #       np.histogram(my_dirver_priors))
      #my_dirver_highest_priors_idx,my_dirver_lowest_priors_idx=\
              #self.findNHighestLowestidx(my_dirver_priors,n=number_of_known_labels)
      #if self.verbose>2: print("driver",driver,"highest and lowest idx: ",
              #my_dirver_highest_priors_idx,my_dirver_lowest_priors_idx)
      #self.y[my_dirver_highest_priors_idx]=1.0
      #self.y[my_dirver_lowest_priors_idx]=0.0
      
      samples_per_driver_idx=range(self.samples_per_driver)
      my_driver_known_temp_vec=np.zeros(number_of_known_samples)
      my_driver_semi_predictions_l=[]
      for i in range(self.ensemble_size):
	      #reset the values for each ensemble member 
	      self.y[:]=-1.0
	      my_driver_known_idx=np.random.choice(samples_per_driver_idx,size=number_of_known_samples,replace=False)
	      temp_rand_vec=np.random.rand(number_of_known_samples)
	      my_driver_logical_vec=my_dirver_scaled_prior[my_driver_known_idx]>temp_rand_vec
	      if self.verbose>1:print("my_driver_logical_vec: ",my_driver_logical_vec);print("(self.y[my_driver_known_idx])[my_driver_logical_vec]",(self.y[my_driver_known_idx])[my_driver_logical_vec])
	      my_driver_known_temp_vec[my_driver_logical_vec]=1.00
	      self.y[my_driver_known_idx]=my_driver_known_temp_vec
	      #other_driver_known_idx=np.random.choice(samples_per_driver_idx,size=number_of_known_other_driver,
	      #				       p=other_dirver_scaled_prob,replace=False)
	      #if self.verbose>2: print("my_driver_known_idx: ",my_driver_known_idx);print("other_driver_known_idx: ",other_driver_known_idx)
	      #self.y[my_driver_known_idx]=1.0
	      #self.y[other_driver_known_idx]=0.0
	      if self.verbose>1: print("my_driver_y: ",self.y);print("my_driver_scaled_prob: ",my_dirver_scaled_prior)
	      if ssa=='lp':
		my_driver_semi_predictions_l.append(self.doLabelPropagation(self.X,self.y,**kwargs)[:,1])
	      if ssa=='ls':
		print("ls shape: ",self.doLabelSpreading(self.X,self.y,**kwargs).shape)
		my_driver_semi_predictions_l.append(self.doLabelSpreading(self.X,self.y,**kwargs)[:,1])
      
      my_driver_semi_predictions=np.vstack(tuple(my_driver_semi_predictions_l))
      my_driver_semi_prediction=np.mean(my_driver_semi_predictions,axis=0)
      if self.verbose>2: print("my_driver_semi_prediction: ",
              my_driver_semi_prediction[0:self.verbose,:])
      #my_dirver_unnormalized_posterior_prob_one=my_driver_semi_prediction[:,1]*my_dirver_priors
      #my_dirver_unnormalized_posterior_prob_zero=my_driver_semi_prediction[:,0]*(1-my_dirver_priors)
      #my_dirver_partition_function=my_dirver_unnormalized_posterior_prob_one\
      #        +my_dirver_unnormalized_posterior_prob_zero
      #porterior_prediction=my_dirver_unnormalized_posterior_prob_one/my_dirver_partition_function
      #return porterior_prediction,\
      return my_driver_semi_prediction,\
              self.all_identifiers[my_driver_f_row:my_driver_l_row]


class ListTools():
  #####following are tools for sorting a list 
  def tryInt(self,s):
	    try:
	       return int(s)
	    except:
	       return s
  def alphaNumkey(self,s):
	    return[ self.tryInt(c) for c in re.split('([0-9]+)',s)]
  def sortNicely(self,l):
	    return sorted(l,key=self.alphaNumkey)


class ReadWriteTools(ListTools):
  
  
  def readCSV2SeprateArray(self,input_name,skip_header=1):
      temp_data=np.genfromtxt(input_name,delimiter=',',skip_header=skip_header,dtype=None)
      return zip(*temp_data)
      
  def readAll2List(self,input_path,folder,skip_header=1,verbose=0):
      all_files_l=listdir(input_path+str(folder))
      all_files_l=self.sortNicely(all_files_l)
      all_data=[]
      my_identifier_l=[]
      if verbose:print(input_path)#,print(all_files_l)
      for my_file in all_files_l:
          my_identifier_l.append(folder+'_'+my_file.replace('.csv',''))
          my_file=input_path+folder+'/'+my_file
          if verbose:print(my_file)
          temp_data=np.genfromtxt(my_file,delimiter=',',skip_header=skip_header,dtype=None)
          all_data.append(temp_data)
          
      return all_data,my_identifier_l

  def readFile2Arr(self,file_name):
      inp_file=open(file_name,'r')
      return np.load(inp_file)


  def writeArr2File(self,data,file_name):
      out_file=open(file_name,'w')
      np.save(out_file,data) 

  def writeResults2CSV(self,all_predictions,all_identifiers,file_name='predictions.csv'):
        ##make a temp result and sort based on it 
        temp_list=[re.sub(r'_\d+','',w) for w in all_identifiers]
        final_results=[(y,z) for (x,y,z) in 
                sorted(zip(temp_list,all_identifiers,all_predictions),key=lambda triple:triple[0])]

        out=open(file_name,'w')
        out.write('driver_trip,prob\n')
        for item in final_results:
            out.write('%s,%s\n'% item)
            
  def sortBasedOnDriver(self,linked_data,all_identifiers):
        temp_list=[re.sub(r'_\d+','',w) for w in all_identifiers]
        sorted_results=[(y,z) for (x,y,z) in 
                sorted(zip(temp_list,linked_data,all_identifiers),key=lambda triple:triple[0])]
	linked_data=np.array([x for (x,y) in sorted_results])
	all_identifiers=np.array([y for (x,y) in sorted_results])
	return linked_data,all_identifiers

######################################################END of CLASSES
def findMinMaxAll(driver_data):
    ft=FeatureTools()
    vel_min_max=[]
    acc_min_max=[]
    for driver in driver_data:
      v_temp=ft.velVec(driver)
      v_abs_temp=ft.vecAbs(v_temp)
      vel_min_max.append(ft.findMinMax(v_abs_temp))
      a_temp=ft.velVec(v_temp)
      a_abs_temp=ft.vecAbs(a_temp)
      acc_min_max.append(ft.findMinMax(a_abs_temp))

    vel_min_max=np.array(vel_min_max)
    acc_min_max=np.array(acc_min_max)
    return [np.max(vel_min_max),np.min(vel_min_max)],[np.max(acc_min_max),np.min(acc_min_max)]

def makeFeaturesWrapper(input_path,driver,vel_max=90,acc_max=45,n=1,verbose=0):

    ft=FeatureTools()
    #test=np.zeros((5,2))
    #test[:,0]=range(5)
    #print("test",test)
    #print(ft.movingAve(test))
    rwt=ReadWriteTools()
    my_driver_features=[]
    my_data_l,my_identifier=rwt.readAll2List(input_path,driver,verbose=verbose)
    for my_data in my_data_l:       
        my_driver_features.append(ft.makeFeatures(my_data,vel_max=vel_max,acc_max=acc_max,n=n))
    return np.array(my_driver_features),np.array(my_identifier)

def makeFeaturesWrapperMapInp(driver):
    #input_path="/Users/danial/Downloads/drivers/"
    #input_path="/home/danial/Downloads/drivers/"
    driver_results=makeFeaturesWrapper(input_path,driver,vel_max=90,acc_max=45,n=1,verbose=0)
    #print("Feature's shape: ", driver_results[0].shape)
    return driver_results

def reduceResult(results):
    all_data=[driver[0] for driver in results]
    all_identifiers=[driver[1]for driver in results]
    all_data=np.vstack(tuple(all_data))
    all_identifiers=np.hstack(tuple(all_identifiers))
    return all_data,all_identifiers

def doLogRegEnsembleWrapper(driver):
    rgm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=500,verbose=0)
    print("Log Reg: ", driver)
    return rgm.doLogRegEnsemble(driver,C=2000)




def doRFEnsembleWrapper(driver):
    rfm=ModelingTools(all_features,all_identifiers,all_weights=all_weights,samples_per_driver=200,
              ensemble_size=500,verbose=0)
    print("RF :", driver)
    return rfm.doRFEnsemble(driver,n_estimators=100,max_features=0.1)

def doGBEnsembleWrapper(driver):
    gbm=ModelingTools(all_features,all_identifiers,all_weights=all_weights,samples_per_driver=200,
              ensemble_size=100,verbose=0)
    print("GB :", driver)
    return gbm.doGBEnsemble(driver,n_estimators=400,max_features='auto',learning_rate=0.1,max_depth=4)




def doRFEnsembleIterWrapper(driver):
    rfm=ModelingTools(all_features,all_identifiers,all_weights=all_weights,ignore_sample_size=10,samples_per_driver=200,
              ensemble_size=500,verbose=0)
    print("RF Iter :", driver)
    return rfm.doRFEnsembleIter(driver,n_estimators=100)

def doADBEnsembleWrapper(driver):
    adbm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=100,verbose=0)
    
    print("ADABOOST :", driver)
    return adbm.doADBEnsemble(driver,n_estimators=1000)

def doSemisupervisedLearning(driver):
    
    sst=SemiSupervisedLearningTools(my_priors,my_priors_identifiers,all_features,
            all_identifiers,samples_per_driver=200,verbose=0,ensemble_size=1)
    return sst.bayesianEstimate(driver,ssa='lp',number_of_known_samples=50,kernel='rbf',
            n_neighbors=1,gamma=1.0,alpha=1.0,max_iter=500)


#####

if __name__=="__main__":
  start=time.time()
  #input_path="/Users/danial/Downloads/drivers/"
  input_path="/home/danial/Downloads/drivers/"

  rwt=ReadWriteTools()
  all_features_file_name='all_features_triplen_501201520bins_accsgn_accdiff_angvel_velacccross_fftvelaccdot_velvelcross_angvel_n1_nosmooth.dat'
  all_features_file_name_pca=all_features_file_name+'.pca40'
  all_identifiers_file_name='all_identifiers_triplen_501201520bins_accsgn_accdiff_angvel_velacccross_fftvelaccdot_velvelcross_angvel_n1_nosmooth.dat'
  
  all_weights_file_name='all_predictions_ensemble_triplen_501201520bins_accsgn_accdiff_angvel_fftvelaccdot_velvelcross_angvel_n1_nosmooth_rf100_ens500_V73.csv'
  all_predictions_file_name='all_predictions_maxfeatureauto_maxdepth4_lrate0.1_400otherdriverscramble_triplen_501201520bins_accsgn_accdiff_angvel_velacccross_fftvelaccdot_velvelcross_angvel_n1_nosmooth_gb_400_ens100_V102.csv'
  #all_priors_file_name='all_predictions_ensemble_2512040bins_accsgn_angvel_fftvelaccdot_velvelcross_n1_nosmooth_rf100_ens500_V64.csv'
  #all_predictions_ssl_file_name='all_predictions_priorV64_ssl_knownl10_knownh50_ls_rbf_gamma0.005_alpha0.85_V65.csv'
  
  calculate_features=True
  calculate_pca=0#True
  calculate_regfit=False
  calculate_rffit=0#True
  calculate_gbfit=True
  random_drivers_sample=0#True
  calculate_rffit_iter=0#True
  calculate_adbfit=0#True
  calculate_ssl=0#True
  all_drivers_l=listdir(input_path)
  print(len(all_drivers_l))
  
  if calculate_features:
      pool=Pool(8)
      results=pool.map(makeFeaturesWrapperMapInp,all_drivers_l)
      pool.close()
      pool.join()
      #makeFeaturesWrapperMapInp(all_drivers_l[0])
      all_features,all_identifiers=reduceResult(results)
      rwt.writeArr2File( all_features,file_name=all_features_file_name)
      rwt.writeArr2File( all_identifiers,file_name=all_identifiers_file_name)
  else:
      print('Reading ...')
      all_features=rwt.readFile2Arr(file_name=all_features_file_name)
      print(all_features.shape)
      #all_features=all_features[:,0:60]
      all_identifiers=rwt.readFile2Arr(file_name=all_identifiers_file_name)
  
  if calculate_pca:
      ft=FeatureTools()
      print('Reading ...')
      all_features=rwt.readFile2Arr(file_name=all_features_file_name)  
      all_features=ft.pcaFit(all_features,n_components=40,verbose=1)
      print(all_features.shape)
      rwt.writeArr2File(all_features,all_features_file_name_pca)
      
  
  print("shape of features and identifiers",all_features.shape,all_identifiers.shape)
  start2=time.time()
  results=[]
  if calculate_regfit:
      rgm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=2,verbose=2)
      driver_number=range(len(all_drivers_l))
      pool=Pool(8)
      results=pool.map(doLogRegEnsembleWrapper,driver_number)
      pool.close()
      pool.join()
      all_predictions,all_identifiers=rgm.reduceResult(results)

  if calculate_rffit:
      rfm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=2,verbose=2)
      all_weights=np.zeros(all_identifiers.shape[0])
      all_weights[:]=2.0
      if random_drivers_sample:
          start_random_sampling=time.time()
          print("doing random_drivers_sample")
          rfm.randomDriverProducer()
          stop_random_sampling=time.time()
          print("total sampling time: ",stop_random_sampling-start_random_sampling)
      print("reading random_drivers_sample")
      other_drivers_random_sampled_file=np.load("exploratory/other_drivers_sample.npz")
      other_drivers_random_sampled=other_drivers_random_sampled_file['results']
      print("finished reading random_drivers_sample; shape:",other_drivers_random_sampled.shape)
      driver_number=range(len(all_drivers_l))
      #test=doRFEnsembleWrapper(driver_number[0])
      pool=Pool(8)
      results=pool.map(doRFEnsembleWrapper,driver_number)
      pool.close()
      pool.join()
      all_predictions,all_identifiers=rfm.reduceResult(results)
      rwt.writeResults2CSV(all_predictions,all_identifiers,file_name=all_predictions_file_name)
      print(all_features.shape,all_predictions.shape)

  if calculate_gbfit:
      gbm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=2,verbose=2)
      all_weights=np.zeros(all_identifiers.shape[0])
      all_weights[:]=2.0
      if random_drivers_sample:
          start_random_sampling=time.time()
          print("doing random_drivers_sample")
          gbm.randomDriverProducer()
          stop_random_sampling=time.time()
          print("total sampling time: ",stop_random_sampling-start_random_sampling)
      print("reading random_drivers_sample")
      other_drivers_random_sampled_file=np.load("exploratory/other_drivers_sample.npz")
      other_drivers_random_sampled=other_drivers_random_sampled_file['results']
      print("finished reading random_drivers_sample; shape:",other_drivers_random_sampled.shape)
      driver_number=range(len(all_drivers_l))
      #test=doRFEnsembleWrapper(driver_number[0])
      pool=Pool(8)
      results=pool.map(doGBEnsembleWrapper,driver_number)
      pool.close()
      pool.join()
      all_predictions,all_identifiers=gbm.reduceResult(results)
      rwt.writeResults2CSV(all_predictions,all_identifiers,file_name=all_predictions_file_name)
      print(all_features.shape,all_predictions.shape)
  


  if calculate_rffit_iter:
      temp,all_weights=rwt.readCSV2SeprateArray(all_weights_file_name)
      all_weights=np.array(all_weights)
      
      all_features,all_identifiers=rwt.sortBasedOnDriver(all_features,all_identifiers)
      rfm=ModelingTools(all_features,all_identifiers,all_weights=all_weights,ignore_sample_size=10,samples_per_driver=200,
              ensemble_size=2,verbose=2)
      driver_number=range(len(all_drivers_l))
      pool=Pool(8)
      results=pool.map(doRFEnsembleIterWrapper,driver_number)
      pool.close()
      pool.join()
      all_predictions,all_identifiers=rfm.reduceResult(results)
      rwt.writeResults2CSV(all_predictions,all_identifiers,file_name=all_predictions_file_name)
      print(all_features.shape,all_predictions.shape)
      #new_weights=(doRFEnsembleIterWrapper(driver_number[3]))
      #new_weights_sortidx=new_weights[0].argsort()
      #print((new_weights[0])[new_weights_sortidx[0:10]],(new_weights[1])[new_weights_sortidx[0:10]]) 
      
  if calculate_adbfit:
      adbm=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              ensemble_size=2,verbose=2)
      driver_number=range(len(all_drivers_l))
      pool=Pool(8)
      results=pool.map(doADBEnsembleWrapper,driver_number)
      pool.close()
      pool.join()
      all_predictions,all_identifiers=adbm.reduceResult(results)
      rwt.writeResults2CSV(all_predictions,all_identifiers,file_name=all_predictions_file_name)
      print(all_features.shape,all_predictions.shape)
  
  if calculate_ssl:
      rwt=ReadWriteTools()
      my_priors_identifiers,my_priors=rwt.readCSV2SeprateArray(all_priors_file_name)
      
      sst=SemiSupervisedLearningTools(my_priors,my_priors_identifiers,all_features,
            all_identifiers,samples_per_driver=200)
      
      driver_number=range(len(all_drivers_l))
      
      pool=Pool(8)
      results=pool.map(doSemisupervisedLearning,driver_number[0:16])
      pool.close()
      pool.join()
      #results=doSemisupervisedLearning(driver_number[0])
      #mt=ModelingTools(all_features,all_identifiers,samples_per_driver=200,
              #ensemble_size=2,verbose=2)
      #all_predictions_ssl,all_identifiers_ssl=mt.reduceResult(results)
      all_predictions_ssl,all_identifiers_ssl=sst.reduceResult(results)
      rwt.writeResults2CSV(all_predictions_ssl,all_identifiers_ssl,
          file_name=all_predictions_ssl_file_name)
 
  stop=time.time()
  print(start2-start)
  print(stop-start2)
  print(stop-start)
  #print(all_identifiers_ssl.shape,all_predictions_ssl.shape)
  #if calculate_rffit:
  
  
  #if calculate_ssl:
    #print(all_identifiers_ssl.shape,all_predictions_ssl.shape)
    #
    

