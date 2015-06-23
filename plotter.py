import numpy as np
import matplotlib.pyplot as plt
from get_cmap import get_cmap, my_color_map
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def generate_Fplot(nStates, states):
  N = len(states)
  F = np.zeros((N,nStates))
  for i,s in enumerate(states):
    for j in range(len(s)):
      F[i,s[j].StateNumber] = 1

  fig = plt.imshow(F,interpolation='nearest')
  plt.ylabel('Time Series')
  plt.xlabel('State')
  plt.show()

def generate_timeseriesplot(data,truestates):
  NumTS = len(data)

  for i in range(NumTS):
    plt.subplot(NumTS,1,i+1)
    plt.plot(data[i])
    y1 = data[i].min()
    y2 = data[i].max()
    count = 0
    for j in range(len(truestates[i])):
      s = truestates[i]
      plt.fill_between(range(count,count+s[j].NumTimePoints), y1, y2, facecolor=my_color_map(s[j].StateNumber))
      count += s[j].NumTimePoints + 1
    plt.axis('off')
  plt.show()

def generate_emissions(theta,data):
  def _eigsorted(cov):
    vals,vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order],vecs[:,order]
  xmin = np.inf; xmax = -np.inf
  ymin = np.inf; ymax = -np.inf
  fig = plt.figure()
  ax = fig.add_subplot(111)
  for i in range(len(data)):
    ax.plot(data[i][:,0],data[i][:,1],'k.')
    xmin = np.min([xmin,np.min(data[i][:,0])])
    xmax = np.max([xmax,np.max(data[i][:,0])])
    ymin = np.min([ymin,np.min(data[i][:,1])])
    ymax = np.max([ymax,np.max(data[i][:,1])])

  vol = [0.25,0.5,0.75,0.95, 0.99]

  ell = []
  for i in range(len(theta)):
    pos = theta[i].mean
    vals,vecs = _eigsorted(theta[i].var)
    th = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    for v in vol:
      width,height = 2.0*np.sqrt(chi2.ppf(v,2))*np.sqrt(vals)
      ell.append(Ellipse(xy=pos,width=width,height=height,angle=th))
  for i,e in enumerate(ell):
    ax.add_artist(e)
    e.set_facecolor(my_color_map(i))
    e.set_alpha(0.5)

  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  plt.show()

if __name__ == '__main__':
  from generate_data import generate_timeseries_set as gtss

  nStates = 4
  dim = 2
  NumTS = 5
  theta, truestates, data = gtss(nStates,dim,NumTS,minl=50, maxl=100)

  generate_Fplot(nStates,truestates)
  generate_timeseriesplot(data,truestates)
  generate_emissions(theta,data)

