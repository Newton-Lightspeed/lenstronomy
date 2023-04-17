import numpy as np
import lenstronomy as lens
import lenstronomy.Util.constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.background import Background
from lenstronomy.Cosmo.nfw_param import NFWParam


class LensCosmo(object):
    def __init__(self, z_lens, z_source, cosmo=None):
        self.z_lens = z_lens
        self.z_source = z_source
        self.background = Background(cosmo=cosmo)
        self.nfw_param = NFWParam(cosmo=cosmo)

    def dd(self):
        return self.background.d_xy(0, self.z_lens)

    def ds(self):
        return self.background.d_xy(0, self.z_source)

    def dds(self):
        return self.background.d_xy(self.z_lens, self.z_source)

    def sis_sigma_v2theta_E(self, v_sigma):
        theta_E = 4 * np.pi * (v_sigma * 1000 / const.c) ** 2 * self.dds() / self.ds() / u.arcsec
        return theta_E

    def sis_theta_E2sigma_v(self, theta_E):
        v_sigma_c2 = theta_E * const.arcsec / (4*np.pi) * self.ds() / self.dds()
        return np.sqrt(v_sigma_c2) * const.c / 1000


class DoubleSourcePlane(object):
    def __init__(self, z_lens, z_source1, z_source2, cosmo=None):
        self.lensCosmo1 = LensCosmo(
            z_lens=z_lens, z_source=z_source1, cosmo=cosmo)
        self.lensCosmo2 = LensCosmo(
            z_lens=z_lens, z_source=z_source2, cosmo=cosmo)

    def einstein_radii_ratio(self, v_sigma):
        theta_E1 = self.lensCosmo1.sis_sigma_v2theta_E(v_sigma)
        theta_E2 = self.lensCosmo2.sis_sigma_v2theta_E(v_sigma)
        return theta_E1 / theta_E2


# set cosmology parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)

# set redshifts
z_lens = 0.5
z_source1 = 1.0
z_source2 = 5.0

# create a double source plane object
double_source = DoubleSourcePlane(z_lens, z_source1, z_source2, cosmo)

# calculate the Einstein radius for a velocity dispersion of 300 km/s
v_sigma = 300
theta_E1 = double_source.lensCosmo1.sis_sigma_v2theta_E(v_sigma)
theta_E2 = double_source.lensCosmo2.sis_sigma_v2theta_E(v_sigma)

# calculate the ratio of Einstein radii
einstein_radii_ratio = double_source.einstein_radii_ratio(v_sigma)

# print the results
print("Angular diameter distances for lens redshift of {:.1f}:".format(z_lens))
print("Dd = {:.2f} Mpc".format(double_source.lensCosmo1.dd()))
print("Ds1 = {:.2f} Mpc".format(double_source.lensCosmo1.ds()))
print("Ds2 = {:.2f} Mpc".format(double_source.lensCosmo2.ds()))
print("Dds1 = {:.2f} Mpc".format(double_source.lensCosmo1.dds()))
print("Dds2 = {:.2f} Mpc".format(double_source.lensCosmo2.dds()))
print("\nEinstein radii for a velocity dispersion of {} km/s:".format(v_sigma))
print("theta_E1 = {:.4f} arcsec".format(theta_E1))
print("theta_E2 = {:.4f} arcsec".format(theta_E2))
print("\nRatio of Einstein radii for the two source planes: {:.4f}".format(einstein_radii_ratio))
