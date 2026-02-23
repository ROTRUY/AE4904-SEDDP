### IMPORTS
from math import pi, exp, log10
import optical_system as OS

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

    def __init__(self, optical_system):
        self.elevation_angle = optical_system['link_benchmark_specification']['average_angle']
        self.D_T = optical_system['link_benchmark_specification']['transmitter_aperture']  # Transmitter aperture
        self.D_R = optical_system['link_benchmark_specification']['receiver_aperture']  # Receiver aperture
        self.Lambda = optical_system['system_specifications']['system_frequency']  # Wavelength
        c = 3e+8  # Speed of light
        self.freq = c / optical_system['system_specifications']['system_frequency']  # Frequency from wavelength
        self.R = optical_system['link_benchmark_specification']['altitude']  # Link range
        self.theta = optical_system['other_specifications']['platform_drift_angle']  # Beam jitter angle
        self.theta_div = optical_system['other_specifications']['beam_width']  # Optical beam divergence

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

    def get_HV57_C_n2(self, h, A=1.7e-14, h_0=0, v=21):
        """
        Refractive index structure constant C_n^2(h) [m^{-2/3}] vs altitude h [m].
        HV57-style model:
        C_n^2(h) = A*exp(h_0/700)*exp(-(h-h_0)/100) + 2.7e-16*exp(-h/1500)
                  + 0.00594*(v/27)^2*(h*1e-5)^10*exp(-h/1000)
        """
        term1 = A * exp(h_0 / 700) * exp(-(h - h_0) / 100)
        term2 = 2.7e-16 * exp(-h / 1500)
        term3 = 0.00594 * (v / 27) ** 2 * (h * 1e-5) ** 10 * exp(-h / 1000)
        return term1 + term2 + term3

    def get_scintillation_index(self):   #equal to the Rytov index
        zenith = 90 - self.elevation_angle
        wave_number = 2 * pi / self.Lambda
        term1 = 2.25 * wave_number **(7/3) * np.secant(zenith) ** (11/6) 
        term2 = self.get_HV57_C_n2(self.R)

        scintillation_index = term1 + term2 
        
        return scintillation_index

optical_system = OS.optical_system1
link_budget = LinkBudget(optical_system)