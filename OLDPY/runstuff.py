import numpy as np
from invwishart import invwishartrand
from sample_truncated_features_init import *
from transformDistStruct import *
from sampleFromMatrixNormal import sampleFromMatrixNormal

import matplotlib.pyplot as plt

class DistStruct():
  def __init__(self):
    self.pi_z = 0
    self.pi_init = 0
    self.pi_s = 0

class DataStruct():
  def __init__(self):
    self.obs = np.asarray([])
    self.labels = np.asarray([])
    self.z_init = np.asarray([])

if __name__ == '__main__':
  # from utilities import *
  # from relabeler import *

  ########################################
  ####         Generate Data          ####
  ########################################
  time_series_dim = 2       # dimension of each time series
  autoRegressive_order = 3  # autoregressive order for each time series
  time_series_len = 1000    # len of each time-series.  Note, each could be a different len.
  m = time_series_dim*autoRegressive_order
  K = np.diag(0.1*np.ones((1,m)))  # matrix normal hyperparameter (affects covariance of matrix)
  M = np.zeros((time_series_dim,m)) # matrix normal hyperparameter (mean matrix)

  nu = time_series_dim + 2   # inverse Wishart degrees of freedom
  meanSigma = 0.5*np.eye(time_series_dim)  # inverse Wishart mean covariance matrix
  nu_delta = (nu-time_series_dim-1)*meanSigma

  numObj    = 5 # number of time series
  numStates = 9 # number of behaviors
  A = []; Sigma = []
  for k in range(numStates):
    Sigma.append(invwishartrand(nu_delta,nu))   # sample a covariance matrix

    if autoRegressive_order == 1:
    # if autoregressive order is 1, use some predefined dynamic matrices that cover the range of stable dynamics
      A.append((-1 +0.2*k)*np.eye(time_series_dim))
    else:
      # otherwise, sample a random set of lag matrices (each behavior might not be very distinguishable!)
      A.append(sampleFromMatrixNormal(M,Sigma[k],K))

  # Define feature matrix by sampling from truncated IBP:
  F = sample_truncated_features_init(numObj,numStates,10)
  # Define transition distributions:
  p_self = 0.95
  pi_z = ((1-p_self)/(numStates-1))*np.ones((numStates,numStates))
  for ii in range(numStates):
    pi_z[ii,ii] = p_self

  pi_init = np.ones((1,numStates))
  pi_init = pi_init/np.sum(pi_init)
  pi_s = np.ones((numStates,1))

  dist_struct_tmp = DistStruct()  ## Build this class!
  dist_struct_tmp.pi_z = pi_z
  dist_struct_tmp.pi_init = pi_init
  dist_struct_tmp.pi_s = pi_s

  data_struct = []

  for nn in range(numObj):
    Kz_inds = (F[nn,:]>0).astype(int)

    pi_z_nn, pi_init_nn = transformDistStruct(dist_struct_tmp,Kz_inds)

    #clear Y X
    labels = np.zeros((time_series_len,),dtype=np.int)
    P = np.cumsum(pi_init_nn)

    labels_temp = np.sum(P[-1]*np.random.rand() > P)
    labels[0] = Kz_inds[labels_temp]

    tmp = np.random.multivariate_normal(np.zeros(time_series_dim),Sigma[labels[0]],autoRegressive_order).T

    x0 = tmp.flatten('F')
    x  = np.atleast_2d(x0).T

    for k in range(time_series_len):
      if k > 0:
        P = np.cumsum(pi_z_nn[labels[k-1],:])
        labels[k] = np.sum(P[-1]*np.random.rand() > P)
      tmp = np.dot(A[labels[k]],x) + \
        np.random.multivariate_normal(np.zeros(time_series_dim),Sigma[labels[k]],1).T
      if k==0:
        Y = tmp
        X = x
      else:
        Y = np.hstack((Y,tmp))
        X = np.hstack((X,x))

      x = np.vstack((tmp,x[:x.shape[0]-time_series_dim,:]))

    #plt.plot(Y.T)
    #plt.plot(labels,'m')
    #plt.show()

    data_struct.append(DataStruct())  ## Build this class!
    data_struct[nn].obs = Y
    data_struct[nn].true_labels = labels

  ##
  ####################################
  ####      Set Model Params     #####
  ####################################

  # Set mean covariance matrix from data (instead of assuming knowledge of
  # ground truth):

  Ybig2 = data_struct[0].obs

  for ii in range(1,len(data_struct)):
    Ybig2 = np.hstack((Ybig2,data_struct[ii].obs))

  nu = time_series_dim + 2
  meanSigma = 0.75*np.cov(np.diff(Ybig2.T))

  obsModelType = 'AR'
  priorType = 'MNIW'

  # Set hyperprior settings for Dirichlet and IBP
  a_alpha   = 1
  b_alpha   = 1
  var_alpha = 1
  a_kappa   = 100
  b_kappa   = 1
  var_kappa = 100
  a_gamma   = 0.1
  b_gamma   = 1

  # The 'getModel' function takes the settings above and creates the
  # necessary 'model' structure.
  assert False
  getModel() ## what should this function return?

  # Setting for inference:
  settings = InferenceSettings() ## Build this class
  settings.Ks = 1  # legacy parameter setting from previous code.  Do not change.
  settings.Niter = 1000  # Number of iterations of the MCMC sampler
  settings.storeEvery = 1  # How often to store MCMC statistics
  settings.saveEvery = 100  # How often to save (to disk) structure containing MCMC sample statistics
  settings.ploton = 1  # Whether or not to plot the mode sequences and feature matrix while running sampler
  settings.plotEvery = 10  # How frequently plots are displayed
  settings.plotpause = 0

  ##

  ####################################
  ######### Run IBP Inference ########
  ####################################

  # Directory to which you want statistics stored.  This directory will be
  # created if it does not already exist:
  settings.saveDir = '../savedStats/BPARHMM/'

  settings.formZInit = 1 # whether or not the sampler should be initialized
                         # with specified mode sequence.  (Experimentally, this seems
                         # to work well.)
  settings.ploton = 1

  # Number of initializations/chains of the MCMC sampler:
  trial_vec = range(10)

  for t in trial_vec:
    z_max = 0
    for seq in range(len(data_struct)):
      # Form initial mode sequences to simply block partition each
      # time series into 'Ninit' features.  Time series are given
      # non-overlapping feature labels:
      T = (data_struct[seq].obs).shape[1]
      Ninit = 5
      init_blocksize = np.floor(T/Ninit)
      z_init = []
      for i in range(Ninit):
        z_init.append(i*np.ones((1,init_blocksize)))
      z_init = np.asarray(z_init)

      z_init[Ninit*init_blocksize+1:T] = Ninit
      data_struct[seq].z_init = z_init + z_max

      z_max = np.max(data_struct[seq].z_init)

    settings.trial = t

    # Call to main function:
    #IBPHMMinference(data_struct,model,settings)  ## what does this file return?
