class ObservationModel()
  def __init__(self):
    self.type = ''
    self.r = 0
    self.priorType = ''
    self.params = []
    self.y_params = YParams(nu_y,).nu = nu_y; #dy + 2;
    model.obsModel.y_params.nu_delta

class HMMparams():
  def __init__(self):
    self.a_alpha   = 0.  # affects \pi_z
    self.b_alpha   = 0.
    self.var_alpha = 0.
    self.a_kappa   = 0.  # affects \pi_z
    self.b_kappa   = 0.
    self.var_kappa = 0.
    self.a_gamma   = 0.  # global expected # of HMM states (affects \beta)
    self.b_gamma   = 0.
    self.harmonic  = 0.
    self.a_sigma   = 0.
    self.b_sigma   = 0.
    self.a_eta     = 0.
    self.b_eta     = 0.

class HiddenMarkovModel():
  def __init__(self):
    self.params = HMMparams()

  def set_parameter_values(numObj,a_alpha,b_alpha,var_alpha,a_kappa,b_kappa,var_kappa,a_gamma,b_gamma):
    self.params.a_alpha   = a_alpha;  # affects \pi_z
    self.params.b_alpha   = b_alpha;
    self.params.var_alpha = var_alpha;
    self.params.a_kappa   = a_kappa;  # affects \pi_z
    self.params.b_kappa   = b_kappa;
    self.params.var_kappa = var_kappa;
    self.params.a_gamma   = a_gamma;  # global expected # of HMM states (affects \beta)
    self.params.b_gamma   = b_gamma;

    self.params.harmonic  = np.sum(1./np.asarray(range(1,numObj+1)))

class ModelClass(self)

  ##
  # Set Hyperparameters

  # Type of dynamical system:
  self.obsModel.type = obsModelType(priorType,)

if strcmp(obsModelType,'AR')
    # Order of AR process:
    model.obsModel.r = r;
    m = d*r;
else
    m = d;
end

# Type of prior on dynamic parameters. Choices include matrix normal
# inverse Wishart on (A,Sigma) and normal on mu ('MNIW-N'), matrix normal
# inverse Wishart on (A,Sigma) with mean forced to 0 ('MNIW'), normal on A,
# inverse Wishart on Sigma, and normal on mu ('N-IW-N'), and fixed A,
# inverse Wishart on Sigma, and normal on mu ('Afixed-IW-N').  NOTE: right
# now, the 'N-IW-N' option is only coded for shared A!!!
model.obsModel.priorType = priorType;

switch model.obsModel.priorType
    case 'NIW'

        model.obsModel.params.M  = zeros([d 1]);
        model.obsModel.params.K =  kappa;
        
    case 'IW-N'
        # Mean and covariance for Gaussian prior on mean:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
    
    case 'MNIW'
        # Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);

        # Inverse covariance along rows of A (sampled Sigma acts as
        # covariance along columns):
        model.obsModel.params.K =  K(1:m,1:m);
        
    case 'MNIW-N'
        # Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);

        # Inverse covariance along rows of A (sampled Sigma acts as
        # covariance along columns):
        model.obsModel.params.K =  K(1:m,1:m);

        # Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));

    case 'N-IW-N'
        # Mean and covariance for A matrix:
        model.obsModel.params.M  = zeros([d m]);
        model.obsModel.params.Lambda0_A = inv(kron(inv(K),meanSigma));

        # Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
        
    case 'Afixed-IW-N'
        # Set fixed A matrix:
        model.obsModel.params.A = A_shared;
        
        # Mean and covariance for mean of process noise:
        model.obsModel.params.mu0 = zeros(d,1);
        model.obsModel.params.cholSigma0 = chol(sig0*eye(d));
        
    case 'ARD'        # Gamma hyperprior parameters for prior on precision parameter:
        model.obsModel.params.a_ARD = a_ARD;
        model.obsModel.params.b_ARD = b_ARD;
        
        # Placeholder for initializeStructs. Can I get rid of this?
        model.obsModel.params.M  = zeros([d m]);

        # Mean and covariance for mean of process noise:
        model.obsModel.params.zeroMean = 1;
end
        
# Degrees of freedom and scale matrix for covariance of process noise:
model.obsModel.params.nu = nu; #d + 2;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-d-1)*meanSigma;

if strcmp(obsModelType,'SLDS')
    # Degrees of freedom and scale matrix for covariance of measurement noise:
    model.obsModel.y_params.nu = nu_y; #dy + 2;
    model.obsModel.y_params.nu_delta = (model.obsModel.y_params.nu-dy-1)*meanR;
    
    model.obsModel.y_priorType = y_priorType;
    
    switch model.obsModel.y_priorType
        case 'NIW'
            
            model.obsModel.y_params.M  = zeros([dy 1]);
            model.obsModel.y_params.K =  kappa_y;
            
        case 'IW-N'
            # Mean and covariance for Gaussian prior on mean:
            model.obsModel.y_params.mu0 = mu0_y; #zeros(dy,1);
            model.obsModel.y_params.cholSigma0 = chol(sig0_y*eye(dy));
    end
    
    # Fixed measurement matrix:
    model.obsModel.params.C = [eye(dy) zeros(dy,d-dy)];
    
    # Initial state covariance:
    model.obsModel.params.P0 = P0*eye(d);
end

# Always using DP mixtures emissions, with single Gaussian forced by
# Ks=1...Need to fix.
model.obsModel.mixtureType = 'infinite';

# Sticky HDP-HMM parameter settings:
model.HMMmodel = HiddenMarkovModel()
model.HMMmodel.set_parameter_values(numObj,a_alpha,b_alpha,var_alpha,a_kappa,b_kappa,var_kappa,a_gamma,b_gamma)

if exist('Ks')
    if Ks>1:
        model.HMMmodel.params.a_sigma = 1;
        model.HMMmodel.params.b_sigma = 0.01;
else:
    Ks = 1;

if exist('Kr'):
    if Kr > 1:
        model.HMMmodel.params.a_eta = 1;
        model.HMMmodel.params.b_eta = 0.01;
    end
end

