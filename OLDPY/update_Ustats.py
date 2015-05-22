def update_Ustats(data_struct,INDS,stateCounts,obsModelType):
  # function Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType)
  Ns = stateCounts.Ns
  Kz, Ks = Ns.shape

  if obsModelType == 'Gaussian':
    dimu = (data_struct[0].obs).shape[0]
    store_YY = np.zeros((dimu,dimu,Kz,Ks))
    store_sumY = np.zeros((dimu,Kz,Ks))
    store_card = np.zeros((Kz,Ks))
    for ii in range(len(data_struct)):
      unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T
      u = data_struct[ii].obs
      kz = unique_z
      unique_s_for_z = np.nonzero(Ns[kz,:,ii])
      ks = unique_s_for_z
      obsInd = INDS[ii].obsIndzs[kz,ks].inds[:INDS[ii].obsIndzs[kz,ks].tot]
      store_YY[:,:,kz,ks] = store_YY[:,:,kz,ks] + np.dot(u[:,obsInd],u[:,obsInd].T)
      store_sumY[:,kz,ks] = store_sumY[:,kz,ks] + np.sum(u[:,obsInd],axis=1)
      store_card = store_card + Ns[:,:,ii]

    Ustats.card = store_card
    Ustats.YY   = store_YY
    Ustats.sumY = store_sumY

  if obsModelType == 'AR' or obsModelType == 'SLDS':
    dimu = (data_struct[0].obs).shape[0]
    dimX = (data_struct[0].X).shape[0]
    store_XX   = np.zeros((dimX,dimX,Kz,Ks))
    store_YX   = np.zeros((dimu,dimX,Kz,Ks))
    store_YY   = np.zeros((dimu,dimu,Kz,Ks))
    store_sumY = np.zeros((dimu,Kz,Ks))
    store_sumX = np.zeros((dimX,Kz,Ks))
    store_card = np.zeros((Kz,Ks))

    for ii in range(len(data_struct)):
      unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T
      u  = data_struct[ii].obs
      X  = data_struct[ii].X
      kz = unique_z
      unique_s_for_z = np.nonzero(Ns[kz,:,ii])
      ks = unique_s_for_z
      obsInd = INDS[ii].obsIndzs[kz,ks].inds[:INDS[ii].obsIndzs[kz,ks].tot]

    store_XX[:,:,kz,ks] = store_XX[:,:,kz,ks] + np.dot(X[:,obsInd],X[:,obsInd].T)
    store_YX[:,:,kz,ks] = store_YX[:,:,kz,ks] + np.dot(u[:,obsInd],X[:,obsInd].T)
    store_YY[:,:,kz,ks] = store_YY[:,:,kz,ks] + np.dot(u[:,obsInd],u[:,obsInd].T)
    store_sumY[:,kz,ks] = store_sumY[:,kz,ks] + np.sum(u[:,obsInd],axis=1)
    store_sumX[:,kz,ks] = store_sumX[:,kz,ks] + np.sum(X[:,obsInd],axis=1)
    store_card = store_card + Ns[:,:,ii]

    Ustats.card = store_card
    Ustats.XX = store_XX
    Ustats.YX = store_YX
    Ustats.YY = store_YY
    Ustats.sumY = store_sumY
    Ustats.sumX = store_sumX

    if obsModelType == 'SLDS' and stateCounts.Nr:  # Don't update if just using z_init
      Nr = stateCounts.Nr
      Kr = len(Nr)
      unique_r = np.nonzero(Nr)
      dimy = (data_struct[0].tildeY).shape[0]
      store_tildeYtildeY = np.zeros((dimy,dimy,Kr))
      store_sumtildeY    = np.zeros((dimy,Kr))
      store_card         = np.zeros((1,Kr))

      for ii in range(len(data_struct)):
        tildeY = data_struct[ii].tildeY
        kr = unique_r
        obsInd_r = INDS[ii].obsIndr[kr].inds[:INDS[ii].obsIndr[kr].tot]
        store_tildeYtildeY[:,:,kr] = store_tildeYtildeY[:,:,kr] + np.dot(tildeY[:,obsInd_r],tildeY[:,obsInd_r].T)
        store_sumtildeY[:,kr] = store_sumtildeY[:,kr] + np.sum(tildeY[:,obsInd_r],axis=1)
        store_card = store_card + Nr[ii,:]

      Ustats.Ustats_r.YY = store_tildeYtildeY
      Ustats.Ustats_r.sumY = store_sumtildeY
      Ustats.Ustats_r.card = store_card

  elif obsModelType == 'Multinomial':
    numVocab = data_struct[0].numVocab
    store_counts = np.zeros((numVocab,Kz,Ks))

    for ii in range(len(data_struct)):
      u = data_struct[ii].obs
      unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T
      kz = unique_z
      unique_s_for_z = np.nonzero(Ns[kz,:,ii])
      ks = unique_s_for_z
      obsInd = INDS[ii].obsIndzs[kz,ks].inds[:INDS[ii].obsIndzs[kz,ks].tot]
      store_counts[:,kz,ks] = store_counts[:,kz,ks] + np.digitize(u[obsInd],range(numVocab)).T
    Ustats.card = store_counts

  else:
    raise ValueError('Error in update_Ustats:  wrong obsModelType')

  return Ustats
