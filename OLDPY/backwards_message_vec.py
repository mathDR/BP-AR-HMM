import numpy as np

def backwards_message_vec(likelihood,blockEnd,pi_z,pi_s):
  #function [bwds_msg, partial_marg] = backwards_message_vec(likelihood,blockEnd,pi_z,pi_s)
  # Allocate storage space
  Kz = pi_z.shape[1]
  Ks = pi_s.shape[1]
  T  = len(blockEnd)

  bwds_msg     = np.ones((Kz,T))
  partial_marg = np.zeros((Kz,T))

  # Compute marginalized likelihoods for all times, integrating s_t
  if Kz==1 and Ks==1:
    marg_like = np.squeeze(likelihood).T
  else:
    marg_like = np.squeeze(np.sum(likelihood * pi_s[:,:,np.ones((1,1,blockEnd[-1]))],axis=1) )


  # If necessary, combine likelihoods within blocks, avoiding underflow
  if T < blockEnd[-1]:
    marg_like = np.log(marg_like+eps)

    block_like = np.zeros((Kz,T))
    block_like[:,0] = np.sum(marg_like[:,:blockEnd[0]],axis=1)
    for tt in range(1,T):
      block_like[:,tt] = np.sum(marg_like[:,blockEnd[tt-1]+1:blockEnd[tt]],axis=1)

    block_norm = np.max(block_like,axis=0)
    block_like = np.exp(block_like - block_norm[np.ones((Kz,1)),:])
  else:
    block_like = marg_like

  # Compute messages backwards in time
  for tt in range(T-1,0,-1):
    # Multiply likelihood by incoming message:
    partial_marg[:,tt+1] = block_like[:,tt+1] * bwds_msg[:,tt+1]
    # Integrate out z_t:
    bwds_msg[:,tt] = np.dot(pi_z , partial_marg[:,tt+1])
    bwds_msg[:,tt] = bwds_msg[:,tt] / np.sum(bwds_msg[:,tt])

  # Compute marginal for first time point
  partial_marg[:,0] = block_like[:,0] * bwds_msg[:,0]

  return bwds_msg, partial_marg
