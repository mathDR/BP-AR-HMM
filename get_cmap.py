import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def my_color_map(N):
    from numpy import mod
    colormap = ['r','g','b','k','c','m','y']
    return colormap[mod(N,7)]

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def main():
    N = 30
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plt.axis('scaled')
    ax.set_xlim([ 0, N])
    ax.set_ylim([-0.5, 0.5])
    cmap = get_cmap(N)
    for i in range(N):
        col = cmap(i)
        rect = plt.Rectangle((i, -0.5), 1, 1, facecolor=col)
        ax.add_artist(rect)
    ax.set_yticks([])
    plt.show()

if __name__=='__main__':
    main()
