#!/usr/local/bin/python3
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt

# To make gif using cluster:
# add matplotlib.use('Agg') to top
# module load python/2.7.12

# used to make anim7.gif:  convert -delay 15 *.png anim7.gif
# two identical (~touching) boxes, no colorbar
#   plus a third box for the projected density distribution histograms
# interaction point in center of plots
# ax1,2 y-range = 2.5 mm, for smaller surface charges

def main():
    
    # for n in range(76, 77, 2):
    for n in range(1, 90, 2):
        if (n < 10):
            fname = 'ed00' + str(n) + '.dat'
        else:
            if (n < 100):
                fname = 'ed0' + str(n) + '.dat'
            else:
                fname = 'ed' + str(n) + '.dat'
        make_png(fname)

def make_png(fname1):

    z_index = 2
    # get max value of z to plot, if required
    max_z = None

    # create a figure with two subplots, ax1 and ax2, one over the other
    # adjust the vertical size and horizontal spacing to get the botton/top x-axes to match up
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                        gridspec_kw={'height_ratios': [0.5, 0.8, 2],
                                                     'hspace': -0.05},
                                        figsize=(7.5, 7))
    #fig.suptitle('Electron/Hole density vs. position\n', fontsize=16, linespacing=0.4)

    # get data for electron density plot
    data = np.loadtxt(fname1)
    x = set(data[:,0])  # sets only have one copy of anything; any repeats are removed
    y = set(data[:,1])
    z = data[:,z_index]

    # reshape the zvals array into the appropriate shape, and find the boundaries
    zvals = z.reshape(len(x), len(y))
    zvals[zvals < 1.1e-12] = 0
    if (max_z != None): zvals[zvals > max_z] = 0

    # imshow plots columns and rows opposite to how you'd expect; so transpose them
    zvals = zvals.T
    bounds = (min(x), max(x), min(y), max(y))

    # now get data for hole density plot
    fname2 = "h" + fname1[1:]
    data = np.loadtxt(fname2)
    x2 = set(data[:,0])
    y2 = set(data[:,1])
    z2 = data[:,z_index]
    zvals2 = z2.reshape(len(x2), len(y2))
    zvals2[zvals2 < 1.1e-12] = 0
    if (max_z != None): zvals2[zvals2 > max_z] = 0
    zvals2 = zvals2.T
    bounds2 = (min(x2), max(x2), min(y2), max(y2))

    # plot the lower image
    ip = ax2.imshow(zvals2,
                    norm=colors.LogNorm(vmax=1.0e3, vmin=1.0e-11),
                    extent=bounds2,   # set the boundaries of the edges of the 'image' data
                    origin="lower",  # tell matplotlib that [0,0] is at the bottom
                    cmap='jet')      # use the 'jet color map scheme, there are a bunch of options
    # plot the upper image
    ip = ax1.imshow(zvals,
                    norm=colors.LogNorm(vmax=1.0e3, vmin=1.0e-11),
                    extent=bounds,   # set the boundaries of the edges of the 'image' data
                    origin="lower",  # tell matplotlib that [0,0] is at the bottom
                    cmap='jet')      # use the 'jet color map scheme, there are a bunch of options
    # see: https://matplotlib.org/examples/color/colormaps_reference.html

    # select a specific zoomed range for the plot
    # share axes so that I need to specify only one set of ranges
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.get_shared_x_axes().join(ax1, ax3)
    ax1.set_xlim(1, 19)
    ax1.set_ylim(0, 2.5)

    # label axes
    plt.setp(ax1, xticklabels=[])
    plt.setp(ax2, xticklabels=[])
    ax1.set_ylabel("Z [mm]", labelpad=8,  size=10)
    ax1.set_title("Electron/Hole density\n", fontsize=16, linespacing=0.3)
    ax3.set_xlabel("Radius [mm]", size=13)
    ax2.set_ylabel("Z [mm]", labelpad=8,  size=10)

    # plot the e/h number projections onto radius
    
    xx = np.arange(len(x))
    xx = xx * max(x)/len(x)
    zz1 = zvals.sum(axis=0)
    zz2 = zvals2.sum(axis=0)
    ax3.semilogy(xx, zz1, '-')
    ax3.semilogy(xx, zz2, '-')
    ax3.set_ylim(0.01, 100)

    # make the color legend
    # cbar = plt.colorbar(ip, fraction=0.046, pad=0.04)

    #plt.tight_layout()
    #plt.show()
    fname3 = "gif_data/eh" + fname1[1:5] + ".png"
    print('Saving frame', fname3)
    plt.savefig(fname3)
    plt.close("all")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


if __name__ == "__main__":
    main()
