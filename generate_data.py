from __future__ import division
import numpy as np

class EmissionStatistics():
  def __init__(self,mean=0.0,var=1.0):
    self.mean = mean
    self.var = var

class StateStatistics():
  def __init__(self,NumTimePoints=0,StateNumber=0):
    self.NumTimePoints = NumTimePoints
    self.StateNumber = StateNumber

def generate_timeseries(theta, dim, num, minl=50, maxl=1000):
  nStates = len(theta)
  partition = np.random.randint(minl, maxl, num)
  if dim == 1:
    data = np.empty([], dtype=np.float64)
  else:
    data = np.empty((1,dim), dtype=np.float64)

  states = []
  oldstate = np.random.randint(nStates)
  for p in partition:
    state = np.random.randint(nStates)
    while state==oldstate:
      state = np.random.randint(nStates) # Guarantees a transition
    oldstate = state
    mean=theta[state].mean
    var=theta[state].var
    if dim==1:
      tdata = np.random.normal(mean, var, p)
    else:
      tdata = np.random.multivariate_normal(mean, var, p)

    data = np.concatenate((data, tdata))
    states.append(StateStatistics(p,state))

  if dim == 1:
    return np.atleast_2d(data).T, states
  else:
    return data[1:], states

def generate_timeseries_set(nStates,dim,N,minl=50, maxl=1000):
  '''Generate a set of N time series each having markov switching dynamics among nStates.  Each time point has dimension dim and each segment length is in [minl,maxl]
  '''
  # Generate statistics for each state
  theta = []
  for s in range(nStates):
    if dim == 1:
      mean = np.random.randn()*10
      var = np.random.randn()*1
      if var < 0:
        var = var * -1
    else:
      mean = np.random.standard_normal(dim)*10
      # Generate a random SPD matrix
      A = np.random.standard_normal((dim,dim))
      var = np.dot(A,A.T)
    theta.append(EmissionStatistics(mean,var))
  # Now theta has length nStates, with emmision parameters for each state
  # Build dataset
  data = []
  truestates = []
  for i in range(N):
    num = np.random.randint(2,nStates,1) # Pick a random number of behaviors for this time series to have
    tdata, tstates = generate_timeseries(theta,dim,num)
    # tdata is raw data, tstates is a structure

    data.append(tdata)
    truestates.append(tstates)
  return theta, truestates, data

if __name__ == '__main__':
  N = 5
  theta, truestates, data = generate_timeseries_set(4,2,N,minl=50, maxl=100)

  assert len(theta)==4
  for i in range(N):
    count = 0
    for j in range(len(truestates[i])):
      count += truestates[i][j].NumTimePoints
      #print truestates[i][j].NumTimePoints, truestates[i][j].StateNumber
    assert len(data[i])==count







