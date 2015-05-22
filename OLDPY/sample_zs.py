import numpy as np
from transformDistStruct import *

def sample_zs(data_struct,dist_struct,F,theta,obsModelType):
  # function [stateSeq INDS stateCounts] = sample_zs(data_struct,dist_struct,F,theta,obsModelType)

  ####################################
  # Define and initialize parameters #
  ####################################

  numObj = len(data_struct)

  Kz = (dist_struct[0].pi_z).shape[1]
  Ks = (dist_struct[0].pi_s).shape[1]

  # Initialize state count matrices:
  N = np.zeros((Kz+1,Kz,numObj))
  Ns = np.zeros((Kz,Ks,numObj))

  # Preallocate INDS
  for ii in range(len(data_struct)):
    T = len(data_struct[ii].blockSize)
    INDS[ii].obsIndzs[:Kz,:Ks] = {'inds':sparse(1,T),'tot':0}
    stateSeq[ii] = {'z':np.zeros((1,T)),'s':np.zeros((1,data_struct[ii].blockEnd[-1]))}

  for ii in range(len(data_struct)):
    # Define parameters:
    pi_z, pi_init = transformDistStruct(dist_struct[ii],F[ii,:])
    pi_s = dist_struct[ii][pi_s]

    T = len(data_struct[ii].blockSize)
    blockSize = data_struct[ii].blockSize
    blockEnd = data_struct[ii].blockEnd

    # Initialize state and sub-state sequences:
    z = np.zeros((1,T))
    s = np.zeros((1,np.sum(blockSize)))

    ####################################
    # Compute likelihoods and messages #
    ####################################

    # Compute likelihood(kz,ks,u_i) of each observation u_i under each
    # parameter theta(kz,ks):
    ## Kz_inds = find(F(ii,:)>0)
    likelihood = compute_likelihood(data_struct[ii],theta,obsModelType,Kz_inds,Kz,Ks)

    # Compute backwards messages:
    [bwds_msg, partial_marg] = backwards_message_vec(likelihood, blockEnd, pi_z, pi_s)

    ############################################
    # Sample the state and sub-state sequences #
    ############################################

    # Sample (z(1),{s(1,1)...s(1,N1)}).  We first sample z(1) given the
    # observations u(1,1)...u(1,N1) having marginalized over the associated s's
    # and then sample s(1,1)...s(1,N1) given z(1) and the observations.

    totSeq = np.zeros((Kz,Ks))
    indSeq = np.zeros((T,Kz,Ks))

    for t in range[t]:
      # Sample z[t]:
      if t == 1:
        Pz = pi_init.T * partial_marg[:,0]
        obsInd = range(blockEnd[0])
      else:
        Pz = pi_z[z[t-1],:].T * partial_marg[:,t]
        obsInd = range(blockEnd[t-1]+1,blockEnd[t])
      
      Pz   = np.cumsum(Pz)
      z[t] = 1 + np.sum(Pz[-1]*np.random.rand() > Pz)

      # Add state to counts matrix:
      if t > 1:
        N[z[t-1],z[t],ii] += 1
      else:
        N[Kz+1,z[t],ii] += 1 # Store initial point in "root" restaurant Kz+1

      # Sample s(t,1)...s(t,Nt) and store sufficient stats:
      for k in range(blockSize[t]):
        # Sample s(t,k):
        if Ks > 1:
          Ps = pi_s[z[t],:] * likelihood[z[t],:,obsInd[k]]
          Ps = np.cumsum(Ps)
          s[obsInd[k]] = 1 + np.sum(Ps[-1]*np.random.rand() > Ps)
        else:
          s[obsInd[k]] = 1

        # Add s(t,k) to count matrix and observation statistics:
        Ns[z[t],s[obsInd[k]],ii] += 1
        totSeq[z[t],s[obsInd[k]]] += 1
        indSeq[totSeq[z[t],s[obsInd[k]]],z[t],s[obsInd[k]]] = obsInd[k]

    stateSeq[ii].z = z
    stateSeq[ii].s = s

    for jj in range(Kz):
      for kk in range(Ks):
        INDS[ii].obsIndzs[jj,kk].tot  = totSeq[jj,kk]
        INDS[ii].obsIndzs[jj,kk].inds = sparse(indSeq[:,jj,kk].T)

  stateCounts.N = N
  stateCounts.Ns = Ns

  return stateSeq, INDS, stateCounts
