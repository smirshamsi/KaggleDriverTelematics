from multiprocessing import Pool
import numpy as np
from os import listdir
import scipy.stats as scs
def readfile(fname):
  return np.loadtxt(fname,delimiter=',')
class Featurs():
  def vecAbs(self,vec):
      return np.sqrt(vec[:,0]**2+vec[:,1]**2)

  def velVec(self,my_data,n=1):
      return np.diff(my_data,n=n,axis=0) 
  
  def accVec(self,my_data,n=1):
      vel_vec_temp=self.velVec(my_data,n=n)
      return self.velVec(vel_vec_temp,n=1)

  def velQuant(self,my_data,,prob=np.arange(0,1.05,0.05),n=1):
      vel_vec_temp=self.velVec(my_data,n=n)
      vel_vec_abs=self.vecAbs(vel_vec_temp)
      return scs.mstats.mquantiles(vel_vec_abs,prob=prob,aphap=1,betap=1,axis=0)

   def accQuant(self,my_data,,prob=np.arange(0,1.05,0.05),n=1):
      acc_vec_temp=self.accVec(my_data,n=n)
      acc_vec_abs=self.accAbs(acc_vec_temp)
      return scs.mstats.mquantiles(acc_vec_abs,prob=prob,aphap=1,betap=1,axis=0)

if __name__=="__main__":
  input_path="/Users/danial/Downloads/drivers/"
  all_drivers_l=listdir(input_path)
  print(type(all_drivers_l))
