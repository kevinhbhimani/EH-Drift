import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import norm
from sklearn import preprocessing
from scipy.stats import norm
import statistics
from scipy.optimize import curve_fit


#directory_20m = '/pscratch/sd/k/kbhimani/siggen_ccd_data/5000.00_keV/grid_0.0200/self_repulsion_1/P42575A/q=0.00/drift_data_r=15.00_z=5.00/'
directory_20m = '/Users/kevinhbhimani/Desktop/GPU_work/test/'

directory_10m = '/pscratch/sd/k/kbhimani/siggen_ccd_data/5000.00_keV/grid_0.0100/self_repulsion_1/P42575A/q=0.00/drift_data_r=15.00_z=5.00/'

i=700
imp_rad=15 #r location of impact
imp_height=5 #z location of impact

r_min=14.5
r_max=15.5

data_e_20m=np.loadtxt(directory_20m+ 'ed{:03d}.dat'.format(i))
data_h_20m=np.loadtxt(directory_20m+ 'hd{:03d}.dat'.format(i))
# data_e_10m=np.loadtxt(directory_10m+ 'ed{:03d}.dat'.format(i))
# data_h_10m=np.loadtxt(directory_10m+ 'hd{:03d}.dat'.format(i))

rad_e_20m=data_e_20m[:,0]
height_e_20m=data_e_20m[:,1]
dens_r_e_20m=data_e_20m[:,2]
rad_h_20m=data_h_20m[:,0]
height_h_20m=data_h_20m[:,1]
dens_r_h_20m=data_h_20m[:,2]
# rad_e_10m=data_e_10m[:,0]
# dens_r_e_10m=data_e_10m[:,2]
# rad_h_10m=data_h_10m[:,0]
# dens_r_h_10m=data_h_10m[:,2]




radius = rad_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)]
frequency = dens_r_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)]

frequency = dens_r_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)]/np.max(dens_r_e_20m[(dens_r_e_20m>0) & (height_e_20m==imp_height)])

plt.plot(radius, frequency, 'o',markersize=2, color='tab:blue', alpha=0.7)

# plt.hist(rad_e_20m, weights=dens_r_e_20m, bins=10000, ec='black',color='tab:blue')


def Norm(x, mu, sigma):
    #y = A*np.exp(-1*B*x**2)
    y=(1/(sigma*np.sqrt(2*np.pi)))*np.exp((-(x-mu)**2)/(2*sigma**2))
    return y

def Gauss(x, a_gaus, b_gaus, c_gaus):
    y = a_gaus*np.exp(-(x-b_gaus)**2/(2*c_gaus**2))
    return y

#parameters, covariance = curve_fit(Norm, radius, frequency,  p0=[imp_rad,statistics.stdev(frequency)], check_finite='true')
parameters, covariance = curve_fit(Gauss, radius, frequency,  p0=[np.max(frequency),imp_rad,statistics.stdev(frequency)], check_finite='true')

fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

fit_y = Gauss(radius, fit_A, fit_B, fit_C)
plt.plot(radius, fit_y, '-', color='tab:red', alpha=0.7)


# mu = imp_rad #statistics.mean(radius*frequency)
# sigma = statistics.stdev(frequency)

# # add a 'best fit' line
# fit_pdf=norm.pdf(rad_e_20m, mu+imp_rad, sigma)
# a_gaus=np.max(dens_r_e_20m)
# b_gaus=imp_rad+mu
# c_gaus=sigma


# fit_gaus=a_gaus*np.exp(-(rad_e_20m-b_gaus)**2/(2*c_gaus**2))


# fit_norm=(1/(sigma*np.sqrt(2*np.pi)))*np.exp((-(radius-mu)**2)/(2*sigma**2))

# plt.plot(radius, fit_norm, color='tab:red', alpha=0.7)


plt.xlabel('Radius (mm)')
plt.ylabel('Normalized density')
plt.title(r'$\mathrm{Histogram\ of\ Density:}\ \mu=%.3f,\ \sigma=%.3f$' %(fit_B, fit_C))
plt.legend(['Data','Fit'])

plt.grid(True)

plt.savefig('/Users/kevinhbhimani/Desktop/GPU_work/test/fit.png')
