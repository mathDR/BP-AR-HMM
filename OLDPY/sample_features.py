import numpy as np

def sample_features_init(numObj,gamma0):
  #function F = sample_features_init(numObj,gamma0)

  F[0,:np.random.poisson(gamma0)] = 1;
  if np.sum(F[0,:]) == 0:
      F[0,0] = 1;

  featureCounts = np.sum(F,axis=0)

  posInds = np.nonzero(np.sum(F,axis=0)>0);
  Kz = posInds[-1];

  for ii in range(1,numObj):
    for kk in range(Kz):
      rho = featureCounts[kk]/ii;
      F[ii,kk] = np.random.rand()>(1-rho);

    F[ii,Kz+1:Kz+np.random.poisson(gamma0/ii)] = 1

    if np.sum(F[ii,:]) == 0:
      F[ii,Kz+1] = 1

    featureCounts = np.sum(F,axis=0)

    posInds = np.nonzero(np.where(featureCounts>0));
    Kz = posInds[-1]

  F = F[:,:Kz]
  return F
