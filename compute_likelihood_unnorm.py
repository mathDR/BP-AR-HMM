import numpy as np

def compute_likelihood_unnorm(data_struct,theta,obsModelType,Kz_inds,Kz,Ks):
  #function log_likelihood =
  #  compute_likelihood_unnorm(data_struct,theta,obsModelType,Kz_inds,Kz,Ks)

  if obsModelType == 'Gaussian':
    invSigma = theta.invSigma
    mu = theta.mu

    dimu, T = (data_struct.obs).shape

    log_likelihood = -np.inf*np.ones((Kz,Ks,T))
    kz = Kz_inds
    for ks in range(Ks):
      cholinvSigma = np.linalg.chol(invSigma[:,:,kz,ks])
      dcholinvSigma = np.diag(cholinvSigma)
      u = np.dot(cholinvSigma*(data_struct.obs - mu[:,kz*np.ones((1,T)),ks]))
      log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(np.log(dcholinvSigma))

  elif obsModelType =='AR' or obsModelType == 'SLDS':
    invSigma = theta.invSigma
    A = theta.A
    X = data_struct.X

    dimu, T = (data_struct.obs).shape

    log_likelihood = -np.inf*np.ones((Kz,Ks,T))
    if theta.mu:
      mu = theta.mu
      kz = Kz_inds
      for ks in range(Ks):
        cholinvSigma = np.linalg.chol(invSigma[:,:,kz,ks])
        dcholinvSigma = np.diag(cholinvSigma)
        u = np.dot(cholinvSigma,(data_struct.obs - np.dot(A[:,:,kz,ks],X)-mu[:,kz*np.ones((1,T)),ks]))
        log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(np.log(dcholinvSigma))
    else:
      kz = Kz_inds
      for ks in range(Ks):
        cholinvSigma = np.linalg.chol(invSigma[:,:,kz,ks])
        dcholinvSigma = np.diag(cholinvSigma)
        u = np.dot(cholinvSigma,(data_struct.obs - np.dot(A[:,:,kz,ks],X)))
        log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(np.log(dcholinvSigma))

  elif obsModelType == 'Multinomial':
    log_likelihood = np.log(theta.p[:,:,data_struct.obs])

  else:
    raise ValueError('Error in compute_likelihood_unnorm: obsModelType not defined')

  return log_likelihood
