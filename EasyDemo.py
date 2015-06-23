import numpy as np
import matplotlib.pyplot as plt

from generate_data import generate_timeseries_set as gtss
from plotter import *

# -------------------------------------------------   CREATE TOY DATA!
if __name__ == '__main__':
  print 'Creating some toy data...\n'
  nStates = 4
  numTS = 5
  dim = 2
  # First, we'll create some toy data
  #   5 sequences, each of length between 50 and 500.
  #   Each sequences selects from 4 behaviors,
  #     and switches among its selected set over time.
  #     We'll use num=4 behaviors, each of which defines a distinct Gaussian
  #     emission distribution (with dim=2 dimensions).
  emissions, trueStates, data  = gtss(nStates, dim, numTS, minl=50, maxl=500)

  # Visualize the raw data time series
  # with background colored by "true" hidden state
  generate_timeseriesplot(data,trueStates)

  # Visualize the "true" generating parameters
  # Feat matrix F (binary numTS x nStates matrix )
  generate_Fplot(nStates,trueStates)

  # Emission parameters theta (Gaussian 2D contours)
  generate_emissions(emissions,data)

  # -------------------------------------------------   RUN MCMC INFERENCE!
  #modelP = {'bpM.gamma', 2};
  #algP   = {'Niter', 100,'HMM.doSampleHypers',0,'BP.doSampleMass',0,'BP.doSampleConc',0};
  # Start out with just one feature for all objects
  #initP  = {'F.nTotal', 1};
  #CH = runBPHMM( data, modelP, {1, 1}, algP, initP );
  # CH is a structure that captures the "Chain History" of the MCMC
  #  it stores both model config at each each iteration (in Psi field)
  #             and diagnostic information (log prob, sampler stats,etc.)


  # ---VISUALIZE RESULTS!
  # Remember: the actual labels of each behavior are irrelevent
  #   so there won't in general be direct match with "ground truth"
  # For example, the true behavior #1 may be inferred behavior #4

  # So we'll need to align recovered parameters (separately at each iter)
  # Let's just look at iter 90 and iter 100

  #Psi90 = CH.Psi( CH.iters.Psi == 90 );
  #alignedPsi90 = alignPsiToTruth_OneToOne( Psi90, data );

  #Psi100 = CH.Psi( CH.iters.Psi == 100 );
  #alignedPsi100 = alignPsiToTruth_OneToOne( Psi100, data );

  # Estimated feature matrix F
  #figure( 'Units', 'normalized', 'Position', [0 0.5 0.5 0.5] );
  #subplot(1,2,1);
  #plotFeatMat( alignedPsi90 );
  #title( 'F (@ iter 90)', 'FontSize', 20 );
  #subplot(1,2,2);
  #plotFeatMat( alignedPsi100 );
  #title( 'F (@ iter 100)', 'FontSize', 20 );

  # Estimated emission parameters
  #figure( 'Units', 'normalized', 'Position', [0.5 0.5 0.5 0.5] );
  #subplot(1,2,1);
  #plotEmissionParams( Psi90 );
  #title( 'Theta (@ iter 90)', 'FontSize', 20 );
  #subplot(1,2,2);
  #plotEmissionParams( Psi100 );
  #title( 'Theta (@ iter 100)', 'FontSize', 20 );

  # Estimated state sequence
  #plotStateSeq( alignedPsi100, [1 3] );
  #set( gcf, 'Units', 'normalized', 'Position', [0.1 0.25 0.75 0.5] );
  #title('Est. Z : Seq 3', 'FontSize', 20 );

