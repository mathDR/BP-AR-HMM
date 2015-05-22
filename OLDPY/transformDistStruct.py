import numpy as np

def transformDistStruct(dist_struct,feature_vec):
  #function [pi_z pi_init] = transformDistStruct(dist_struct,feature_vec)
  Kz      = len(feature_vec)
  pi_z    = np.dot(dist_struct.pi_z,np.tile(feature_vec,(Kz,1)))
  pi_z    = pi_z/np.tile(np.sum(pi_z,axis=1),(Kz,1)) # this differs from matlab since the sum returns a (Kz,) shaped array
  pi_init = dist_struct.pi_init*feature_vec
  pi_init = pi_init/np.sum(pi_init)

  return pi_z, pi_init
