import numpy as np

def sample_truncated_features_init(numObj,Kz,gamma0):
  #function F = sample_truncated_features_init(numObj,Kz,gamma0)
  F = np.zeros((numObj,Kz))
  featureCounts = np.sum(F,axis=0)
  Fik_prev = 0
  coeff = 1.0*gamma0/Kz

  for ii in range(numObj):
    for kk in range(Kz):
      rho = (featureCounts[kk] + coeff)/(1.0 + ii + coeff)
      if rho>1:
        F[ii,kk] = 1-Fik_prev
      else:
        sample_set = [Fik_prev, 1-Fik_prev]
        ind = 1+np.random.rand() > (1-rho)
        F[ii,kk] = sample_set[ind]

      F[ii,kk] = np.random.rand() > (1-rho)

      featureCounts[kk] = featureCounts[kk]+F[ii,kk]

  return F

if __name__ == '__main__':
  np.random.seed(1)
  numObj = 5
  Kz = 9
  gamma0 = 10
  F = sample_truncated_features_init(numObj,Kz,gamma0)
  G = np.asarray([[ 1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
 [ 0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.],
 [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.],
 [ 0.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.],
 [ 1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,  0.]])
  try:
    assert(np.all(np.equal(F,G)))
  except AssertionError:
    raise AssertionError('Error in sample_truncated_features, wrong matrix produced')
