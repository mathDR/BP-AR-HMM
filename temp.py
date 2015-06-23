from pylab import figure, show, rand
from matplotlib.patches import Ellipse
import numpy as np

NUM = 1

#ells = [Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360) for i in range(NUM)]

pos = np.asarray([ 6.85495169, -3.16585795])
ells = [Ellipse(xy=pos, width=4.24303987521, height=1.67456874298, angle=77.066706516)]
#print ells
fig = figure()
ax = fig.add_subplot(111, aspect='equal')
for e in ells:
    ax.add_artist(e)
#    e.set_clip_box(ax.bbox)
#    e.set_alpha(rand())
#    e.set_facecolor(rand(3))
#ax.add_artist(ells)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

show()

