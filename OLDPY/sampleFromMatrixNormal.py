import numpy as np
from matrixNormalToNormal import *

def sampleFromMatrixNormal(M,sqrtV,sqrtinvK,nSamples=1):
  # function S = sampleFromMatrixNormal(M,sqrtV,sqrtinvK,nSamples)

  mu, sqrtsigma = matrixNormalToNormal(M,sqrtV,sqrtinvK)

  S = mu + np.dot(sqrtsigma.T,np.random.standard_normal((len(mu),1)))
  S = np.reshape(S,M.shape)
  return S

if __name__ == '__main__':
    np.random.seed(1)
    F = sampleFromMatrixNormal(np.random.random((1,3)),np.random.random((1,1)),np.random.random((1,)),1)
    print F.shape
