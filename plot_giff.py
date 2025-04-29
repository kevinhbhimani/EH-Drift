#Program used to create the files for EH-Drift snapshots, which can be used to create giffs

#!/usr/local/bin/python3

import os
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def main():
    for n in range(1, 200):
        fname = f'ed{n:03}.dat'
        make_png(fname)
        # break  # Uncomment this line for testing

def make_png(fname1):
    # Initialization Variables
    det_plot = 'P00698A'
    r_int = 15.00
    z_int = 0.02
    sc_int = -0.30    # e.g., 0.30
    eng_int = 5000.00
    grid_int = 0.0200
    sd_init = 0.0010

    # Corresponding hole density file
    fname2 = "h" + fname1[1:]  # e.g., if fname1=ed001.dat, fname2=hd001.dat
    
    #Point to location of saved densities
    file_load = (
        f'/work/users/k/b/kbhimani/siggen_ccd_data/'
        f'density_det={det_plot}_r={r_int:.2f}_z={z_int:.2f}_eng={eng_int:.2f}_'
        f'sc={sc_int:.2f}_sd={sd_init:.4f}_grid={grid_int:.4f}/'
    )

    # For saving output frames (GIF frames)
    file_save_base = '/nas/longleaf/home/kbhimani/siggen_ccd/giff_data/'
    dir_name = (
        f'density_det={det_plot}_r={r_int:.2f}_z={z_int:.2f}_eng={eng_int:.2f}_'
        f'sc={sc_int:.2f}_sd={sd_init:.4f}_grid={grid_int:.4f}/'
    )
    file_save = os.path.join(file_save_base, dir_name)

    if not os.path.exists(file_save):
        os.makedirs(file_save)

    # Optional Plot Title
    plot_title = (
        f"Event at r={r_int:.2f} mm, z={z_int:.2f} mm, "
        f"sc={sc_int:.2f}, E={eng_int:.2f} keV, grid={grid_int:.4f}"
    )

    # Plot settings
    r_1, r_2, z_1, z_2 = 3, 28, 0, 2.5
    fig_x, fig_y = 12, 8
    z_index = 2

    # Load data for electron density
    data_e = np.loadtxt(file_load + fname1)
    x = np.unique(data_e[:, 0])
    y = np.unique(data_e[:, 1])
    z_e = data_e[:, z_index]
    zvals_e = z_e.reshape(len(x), len(y))
    zvals_e[zvals_e < 1.1e-15] = 0
    zvals_e = zvals_e.T

    # Load data for hole density
    data_h = np.loadtxt(file_load + fname2)
    x2 = np.unique(data_h[:, 0])
    y2 = np.unique(data_h[:, 1])
    z_h = data_h[:, z_index]
    zvals_h = z_h.reshape(len(x2), len(y2))
    zvals_h[zvals_h < 1.1e-15] = 0
    zvals_h = zvals_h.T

    fig = plt.figure(figsize=(fig_x, fig_y))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    norm = colors.LogNorm(vmin=1.0e-15, vmax=1.0e3)
    cmap = 'jet'

    im1 = ax1.pcolormesh(x, y, zvals_e, norm=norm, cmap=cmap, shading='auto')
    im2 = ax2.pcolormesh(x2, y2, zvals_h, norm=norm, cmap=cmap, shading='auto')

    ax1.set_xlim(r_1, r_2)
    ax1.set_ylim(z_1, z_2)
    ax2.set_xlim(r_1, r_2)
    ax2.set_ylim(z_1, z_2)
    ax3.set_xlim(r_1, r_2)

    # Remove x labels from top and middle
    ax1.tick_params(axis='x', bottom=False, labelbottom=False)
    ax2.tick_params(axis='x', bottom=False, labelbottom=False)

    # Set axis labels
    ax1.set_ylabel("Height (mm)", fontsize=14)
    ax2.set_ylabel("Height (mm)", fontsize=14)
    ax3.set_xlabel("Radius (mm)", fontsize=14)
    ax3.set_ylabel("Projected density", fontsize=14)

    # Optionally re-add title if desired
    # ax1.set_title(plot_title, fontsize=14)

    ax1.text(0.95, 0.85, 'Electrons', ha='right', va='top',
             transform=ax1.transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.6))
    ax2.text(0.12, 0.85, 'Holes', ha='right', va='top',
             transform=ax2.transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.6))

    # Sum projection along one dimension
    xx = np.arange(len(x)) * max(x) / len(x)
    zz_e = zvals_e.sum(axis=0)
    zz_h = zvals_h.sum(axis=0)
    ax3.semilogy(xx, zz_e, '-', label='Electrons')
    ax3.semilogy(xx, zz_h, '-', label='Holes')
    ax3.set_ylim(0.01, 1000)
    ax3.legend(fontsize=14)

    # Single colorbar
    cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
    fig.colorbar(im1, cax=cbar_ax)

    # Build final output filename
    fname_out = os.path.join(
        file_save,
        f"frame{fname1[1:5]}_det={det_plot}_r={r_int:.2f}_z={z_int:.2f}_sc={sc_int:.2f}_"
        f"E={eng_int:.2f}_grid={grid_int:.4f}_sd={sd_init:.4f}.png"
    )

    print("Saving frame:", fname_out)
    plt.savefig(fname_out, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

if __name__ == "__main__":
    main()
