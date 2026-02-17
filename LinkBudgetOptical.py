### IMPORTS
from math import pi, exp, log10

class LinkBudget():
    """
    Docstring for LinkBudget

    :param D_T: Transmitter aperture
    :param D_R: Receiver aperture
    :param Lambda: Wavelength
    :param R: Link range
    :param theta: Beam jitter angle
    :param theta_div: Optical beam divergence
    """

    def __init__(self, P_TX, D_T, D_R, Lambda, R, theta, theta_div):
        self.D_T = D_T  # Transmitter aperture
        self.D_R = D_R  # Receiver aperture
        self.Lambda = Lambda  # Wavelength
        c = 3e+8  # Speed of light
        self.freq = c / Lambda  # Frequency from wavelength
        self.R = R  # Link range
        self.theta = theta  # Beam jitter angle
        self.theta_div = theta_div  # Optical beam divergence

    def dB(self, x):
        return 10 * log10(x)

    def G_T(self):
        """
        Gain of transmitting aperture
        """
        return (pi * self.D_T / self.Lambda)**2
    
    def L_PT(self):
        """
        Pointing loss of the transmitter (assuming a Gaussian-shaped single-mode beam)
        """
        return exp(-8 * self.theta**2 / self.theta_div**2)
    
    def L_FS(self):
        """
        Free-space propagation loss
        """
        return (4 * pi * self.R / self.Lambda)**2
    
    def G_R(self):
        """
        Gain of the receiving aperture
        """
        return 