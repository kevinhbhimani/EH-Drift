import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import statistics
from scipy.optimize import curve_fit


directory_20m = '/pscratch/sd/k/kbhimani/siggen_ccd_data/5000.00_keV_diff_only/grid_0.0200/self_repulsion_0/P42575A/q=0.00/drift_data_r=15.00_z=5.00/'
directory_10m = '/pscratch/sd/k/kbhimani/siggen_ccd_data/5000.00_keV_diff_only/grid_0.0100/self_repulsion_0/P42575A/q=0.00/drift_data_r=15.00_z=5.00/'
r_min=14.5
r_max=15.5
imp_rad=15 #r location of impact
imp_height=5 #z location of impact

def Gauss(x, a_gaus, b_gaus, c_gaus):
    y = a_gaus*np.exp(-(x-b_gaus)**2/(2*c_gaus**2))
    return y

for i in tqdm(range(0,700)):

    data_e_20m=np.loadtxt(directory_20m+ 'ed{:03d}.dat'.format(i))
    data_h_20m=np.loadtxt(directory_20m+ 'hd{:03d}.dat'.format(i))
    data_e_10m=np.loadtxt(directory_10m+ 'ed{:03d}.dat'.format(i))
    data_h_10m=np.loadtxt(directory_10m+ 'hd{:03d}.dat'.format(i))



    rad_e_20m=data_e_20m[:,0]
    height_e_20m=data_e_20m[:,1]
    dens_r_e_20m=data_e_20m[:,2]
    rad_h_20m=data_h_20m[:,0]
    height_h_20m=data_h_20m[:,1]
    dens_r_h_20m=data_h_20m[:,2]
    rad_e_10m=data_e_10m[:,0]
    height_e_10m=data_e_10m[:,1]
    dens_r_e_10m=data_e_10m[:,2]
    rad_h_10m=data_h_10m[:,0]
    height_h_10m=data_h_10m[:,1]
    dens_r_h_10m=data_h_10m[:,2]

    diff_rad_e_20m = rad_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)]
    diff_den_r_e_20m = dens_r_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)]
    diff_den_z_e_20m = dens_r_e_20m[(dens_r_e_20m>0) & (rad_e_20m==imp_rad)]
    diff_h_e_20m = height_e_20m[(dens_r_e_20m>0) & (rad_e_20m==imp_rad)]

    diff_rad_h_20m = rad_h_20m[(dens_r_h_20m>0) & (height_h_20m==imp_height)]
    diff_den_r_h_20m = dens_r_h_20m[(dens_r_h_20m>0) & (height_h_20m==imp_height)]
    diff_den_z_h_20m = dens_r_h_20m[(dens_r_h_20m>0) & (rad_h_20m==imp_rad)]
    diff_h_h_20m = height_h_20m[(dens_r_h_20m>0) & (rad_h_20m==imp_rad)]

    diff_rad_e_10m = rad_e_10m[(dens_r_e_10m>0) & (height_e_10m==imp_height)]
    diff_den_r_e_10m = dens_r_e_10m[(dens_r_e_10m>0) & (height_e_10m==imp_height)]
    diff_den_z_e_10m = dens_r_e_10m[(dens_r_e_10m>0) & (rad_e_10m==imp_rad)]
    diff_h_e_10m = height_e_10m[(dens_r_e_10m>0) & (rad_e_10m==imp_rad)]

    diff_rad_h_10m = rad_h_10m[(dens_r_h_10m>0) & (height_h_10m==imp_height)]
    diff_den_r_h_10m = dens_r_h_10m[(dens_r_h_10m>0) & (height_h_10m==imp_height)]
    diff_den_z_h_10m = dens_r_h_10m[(dens_r_h_10m>0) & (rad_e_10m==imp_rad)]
    diff_h_h_10m = height_h_10m[(dens_r_h_10m>0) & (rad_h_10m==imp_rad)]

    norm_den_r_e_10m = diff_den_r_e_10m/np.max(diff_den_r_e_10m)
    norm_den_r_e_20m = diff_den_r_e_20m/np.max(diff_den_r_e_20m)

    norm_den_r_h_10m = diff_den_r_h_10m/np.max(diff_den_r_h_10m)
    norm_den_r_h_20m = diff_den_r_h_20m/np.max(diff_den_r_h_20m)

    norm_den_z_e_10m = diff_den_z_e_10m/np.max(diff_den_z_e_10m)
    norm_den_z_e_20m = diff_den_z_e_20m/np.max(diff_den_z_e_20m)

    norm_den_z_h_10m = diff_den_z_h_10m/np.max(diff_den_z_h_10m)
    norm_den_z_h_20m = diff_den_z_h_20m/np.max(diff_den_z_h_20m)


    # parameters_e_10m, covariance_e_10m = curve_fit(Gauss, diff_rad_e_10m, norm_den_r_e_10m,  p0=[np.max(norm_den_r_e_10m),imp_rad,statistics.stdev(norm_den_r_e_10m)], check_finite='true')
    # parameters_e_20m, covariance_e_20m = curve_fit(Gauss, diff_rad_e_20m, norm_den_r_e_20m,  p0=[np.max(norm_den_r_e_20m),imp_rad,statistics.stdev(norm_den_r_e_20m)], check_finite='true')

    # plt.plot(diff_rad_e_10m, Gauss(diff_rad_e_10m, parameters_e_10m[0], parameters_e_10m[1], parameters_e_10m[2]), '-', color='tab:green', alpha=0.7)
    # plt.plot(diff_rad_e_20m, Gauss(diff_rad_e_20m, parameters_e_20m[0], parameters_e_20m[1], parameters_e_20m[2]), '-', color='tab:purple', alpha=0.7)


    plt.plot(diff_rad_e_10m, norm_den_r_e_10m, '-', color='tab:green', alpha=0.7)
    plt.plot(diff_rad_e_20m, norm_den_r_e_20m, '-', color='tab:purple', alpha=0.7)
    plt.xlabel('Radius (mm)')
    plt.ylabel('Normalized Density')
    plt.legend(['10 micron','20 micron'])
    plt.title('Electron density')

    fname_e_r = "/pscratch/sd/k/kbhimani/siggen_ccd_data/gif_data/diff_r/electrons/diff_" + 'ed{:03d}.dat'.format(i) + ".png"
    plt.savefig(fname_e_r)
    plt.clf()

    plt.plot(diff_h_e_10m, norm_den_z_e_10m, '-', color='tab:green', alpha=0.7)
    plt.plot(diff_h_e_20m, norm_den_z_e_20m, '-', color='tab:purple', alpha=0.7)
    plt.xlabel('Height (mm)')
    plt.ylabel('Normalized Density')
    plt.legend(['10 micron','20 micron'])
    plt.title('Electron density')

    fname_e_z = "/pscratch/sd/k/kbhimani/siggen_ccd_data/gif_data/diff_z/electrons/diff_" + 'ed{:03d}.dat'.format(i) + ".png"
    plt.savefig(fname_e_z)
    plt.clf()
    


    # parameters_h_10m, covariance_h_10m = curve_fit(Gauss, diff_rad_h_10m, norm_den_r_h_10m,  p0=[np.max(norm_den_r_h_10m),imp_rad,statistics.stdev(norm_den_r_h_10m)], check_finite='true')
    # parameters_h_20m, covariance_h_20m = curve_fit(Gauss, diff_rad_h_20m, norm_den_r_h_20m,  p0=[np.max(norm_den_r_h_20m),imp_rad,statistics.stdev(norm_den_r_h_20m)], check_finite='true')

    # plt.plot(diff_rad_h_10m, Gauss(diff_rad_h_10m, parameters_h_10m[0], parameters_h_10m[1], parameters_h_10m[2]), '-', color='tab:green', alpha=0.7)
    # plt.plot(diff_rad_h_20m, Gauss(diff_rad_h_20m, parameters_h_20m[0], parameters_h_20m[1], parameters_h_20m[2]), '-', color='tab:purple', alpha=0.7)


    plt.plot(diff_rad_h_10m, norm_den_r_h_10m, '-', color='tab:green', alpha=0.7)
    plt.plot(diff_rad_h_20m, norm_den_r_h_20m, '-', color='tab:purple', alpha=0.7)
    plt.xlabel('Radius (mm)')
    plt.ylabel('Normalized Density')
    plt.legend(['10 micron','20 micron'])
    plt.title('Hole density')

    fname_h_r = "/pscratch/sd/k/kbhimani/siggen_ccd_data/gif_data/diff_r/holes/diff_" + 'hd{:03d}.dat'.format(i) + ".png"
    plt.savefig(fname_h_r)
    plt.clf()

    plt.plot(diff_h_h_10m, norm_den_z_h_10m, '-', color='tab:green', alpha=0.7)
    plt.plot(diff_h_h_20m, norm_den_z_h_20m, '-', color='tab:purple', alpha=0.7)
    plt.xlabel('Height (mm)')
    plt.ylabel('Normalized Density')
    plt.legend(['10 micron','20 micron'])
    plt.title('Hole density')

    fname_h_z = "/pscratch/sd/k/kbhimani/siggen_ccd_data/gif_data/diff_z/holes/diff_" + 'hd{:03d}.dat'.format(i) + ".png"
    plt.savefig(fname_h_z)
    plt.clf()
