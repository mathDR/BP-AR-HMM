# Expose the different parameter classes (they are really just structures)

class MCMCParamsBPHMM():
  ''' Creates a struct encoding the default settings for MCMC inference '''
  def __init__(self):
    self.Niter = 50

    self.doSampleFShared = 1
    self.doSampleFUnique = 1
    self.doSampleUniqueZ = 0
    self.doSplitMerge = 0
    self.doSplitMergeRGS = 0
    self.doSMNoQRev = 0 # ignore proposal prob in accept ratio... not valid!

    SM = ()
    #self.SM.featSelectDistr = 'splitBias+margLik'
    #self.SM.doSeqUpdateThetaHatOnMerge = 0

    self.nSMTrials = 5
    self.nSweepsRGS = 5

    self.doSampleZ = 1
    self.doSampleTheta = 1
    self.doSampleEta   = 1

    self.BP = ()
    #self.BP.doSampleMass = 1
    #self.BP.doSampleConc = 1
    #self.BP.Niter = 10
    #self.BP.var_c = 2

    self.HMM = ()
    #self.HMM.doSampleHypers = 1
    #self.HMM.Niter = 20
    #self.HMM.var_alpha = 2
    #self.HMM.var_kappa = 10

    self.RJ = ()
    # Reversible Jump Proposal settings
    #self.RJ.doHastingsFactor = 1
    #self.RJ.birthPropDistr = 'DataDriven'
    #self.RJ.minW = 15
    #self.RJ.maxW = 100

    self.doAnneal = 0
    self.Anneal = ()
    #self.Anneal.T0 = 100
    #self.Anneal.Tf = 10000

class HMMPrior():
  def __init__(self,a_alpha,b_alpha,a_kappa,b_kappa):
    self.a_alpha = a_alpha
    self.b_alpha = b_alpha
    self.a_kappa = a_kappa
    self.b_kappa = b_kappa

class HMMModel():
  def __init__(self,alpha,kappa,prior):
    self.alpha = alpha
    self.kappa = kappa
    self.prior = prior

class BPPrior():
  def __init__(self,a_mass,b_mass,a_conc,b_conc):
    self.a_mass = a_mass
    self.b_mass = b_mass
    self.a_conc = a_conc
    self.b_conc = b_conc

class BPModel():
  def __init__(self,gamma,c,prior):
    self.gamma = gamma
    self.c = c
    self.prior = prior

class ModelParams_BPHMM():
  def __init__(self,data):
    if data.obsType == 'Gaussian':
      self.obsM = obsModel(precMu=1,degFree=3,doEmpCovScalePrior=0,Scoef=1)

    elif data.obsType == 'AR':
      self.obsM = obsModel(doEmpCov=0,doEmpCovFirstDiff=1,degFree=1,Scoef=0.5)
    else:
      raise Error('BPARHMM only supports Gaussian and AutoRegressive Model Observation Types')

    # ------------------------------- HMM params
    prior = HMMPrior(a_alpha=0.01,b_alpha=0.01,a_kappa=0.01,b_kappa=0.01)
    self.hmmM = HMMModel(alpha=1,kappa=25,prior=prior)

    # ================================================== BETA PROCESS MODEL
    # GAMMA: Mass param for IBP, c0   : Concentration param for IBP
    prior = BPPrior(a_mass=0.01,b_mass=0.01,a_conc=0.01,b_conc=0.01)
    self.bpM = BPModel(gamma=5,c=1,prior=prior)


'''class OutputParams_BPHMM( outParams, algP ):
  def __init__(self,outParams,algP):
    saveDir = getUserSpecifiedPath( 'SimulationResults' )

  for aa in range(len(outParams)):
    if aa == 1:
      jobID = force2double(  outParams{aa} )
    elif aa == 2:
      taskID = force2double( outParams{aa} )

    self.jobID = jobID
    self.taskID = taskID
    self.saveDir = fullfile( saveDir, num2str(jobID), num2str(taskID) )
  #if ~exist(     self.saveDir, 'dir' )
  #  [~,~] = mkdir(     self.saveDir )
  #end

  if isfield( algP, 'TimeLimit' ) && ~isempty( algP.TimeLimit ):
    TL = algP.TimeLimit
    Niter = Inf
  else:
    TL = Inf
    Niter = algP.Niter

  if TL <= 5*60 or Niter <= 200:
    self.saveEvery = 5
    self.printEvery = 5
    self.logPrEvery = 1
    self.statsEvery = 1
  elif TL <= 2*3600 or Niter <= 5000
    self.saveEvery = 25
    self.printEvery = 25
    self.logPrEvery = 5
    self.statsEvery = 5
  else:
    self.saveEvery = 50
    self.printEvery = 50
    self.logPrEvery = 10
    self.statsEvery = 10

  self.doPrintHeaderInfo = 1
'''

