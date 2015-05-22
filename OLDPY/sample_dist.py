import numpy as np

def sample_dist(stateCounts,hyperparams,Kextra):
  #function dist_struct = sample_dist(stateCounts,hyperparams,Kextra)

  numObj  = (stateCounts.Ns).shape[2]
  Kz_prev = (stateCounts.Ns).shape[0]
  Kz      = Kz_prev + Kextra
  Ks      = (stateCounts.Ns).shape[1]

  # Define alpha0 and kappa0 in terms of alpha0+kappa0 and rho0:
  alpha0 = hyperparams.alpha0
  kappa0 = hyperparams.kappa0
  sigma0 = hyperparams.sigma0

  N  = stateCounts.N  # N(i,j) = # z_t = i to z_{t+1}=j transitions. N(Kz+1,i) = 1 for i=z_1.
  Ns = stateCounts.Ns  # Ns(i,j) = # s_t = j given z_t=i

  dist_struct[:numObj] = {'pi_z':np.zeros((Kz,Kz)),'pi_init':np.zeros((1,Kz)),'pi_s':np.zeros((Kz,Ks))}

  beta_vec = np.ones((1,Kz))

  Ntemp  = np.zeros((Kz+1,Kz))
  Nstemp = np.zeros((Kz,Ks))

  for ii in range(numObj):

    Ntemp[:Kz_prev,:Kz_prev] = N[:Kz_prev,:,ii]
    Ntemp[-1,1:Kz_prev]      = N[Kz_prev+1,:,ii]
    Nstemp[:Kz_prev,:]       = Ns[:,:,ii]

    if Ks>1:
        # Sample HMM-state-specific mixture weights \psi_j's with truncation
        # level Ks given sampled s stats Ns:
        sigma_vec = (sigma0/Ks)*np.ones((1,Ks))
    else:
        sigma_vec = sigma0

    pi_z = np.zeros((Kz,Kz))
    pi_s = np.zeros((Kz,Ks))
    for j in range(Kz):
      kappa_vec = np.zeros((1,Kz))
      # Add an amount \kappa to Dirichlet parameter corresponding to a
      # self-transition:
      kappa_vec[j] = kappa0
      # Sample \pi_j's given sampled \beta_vec and counts N, where
      # DP(\alpha+\kappa,(\alpha\beta+\kappa\delta[j])/(\alpha+\kappa)) is
      # Dirichlet distributed over the finite partition defined by beta_vec:
      pi_z[j,:] = randdirichlet_unnorm([alpha0*beta_vec + kappa_vec + Ntemp[j,:]].T).T
      # Sample HMM-state-specific mixture weights \psi_j's with truncation
      # level Ks given sampled s stats Ns:
      pi_s[j,:] = randdirichlet([Nstemp[j,:] + sigma_vec].T).T

    pi_init = randdirichlet_unnorm([alpha0*beta_vec + Ntemp[Kz+1,:]].T).T

    if stateCounts.Nr:
      Nr = stateCounts.Nr[ii,:]  # Nr(i) = # r_t = i
      Kr = len(Nr)
      eta0 = hyperparams.eta0
      dist_struct[ii].pi_r = randdirichlet((Nr + eta0/Kr).T).T

    dist_struct[ii].pi_z    = pi_z
    dist_struct[ii].pi_init = pi_init
    dist_struct[ii].pi_s    = pi_s

  return dist_struct
