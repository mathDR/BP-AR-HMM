from numpy import kron
def matrixNormalToNormal(M,sqrtV,sqrtinvK):
  # function [mu,sqrtsigma] = matrixNormalToNormal(M,sqrtV,sqrtinvK)
  # Converts the parameters for a matrix normal A ~ MN(M,V,K)
  # into a  multivariate normal  A(:) ~ N(mu,sigma)

  mu = M[:]
  sqrtsigma = kron(sqrtinvK,sqrtV)

  return mu, sqrtsigma
