### IMPORTS
from math import pi, exp, log10
import numpy as np
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
        self.elevation_angle = optical_system['link_benchmark_specs']['average_angle']
        self.D_T = optical_system['link_benchmark_specs']['transmitter_aperture']  # Transmitter aperture
        self.D_R = optical_system['link_benchmark_specs']['receiver_aperture']  # Receiver aperture
        self.Lambda = optical_system['system_specs']['system_frequency']  # Wavelength
        c = 3e+8  # Speed of light
        self.freq = c / optical_system['system_specs']['system_frequency']  # Frequency from wavelength
        self.h = optical_system['link_benchmark_specs']['altitude']  # Link range
        self.theta = optical_system['other_specs']['platform_drift_angle']  # Beam jitter angle
        self.theta_div = optical_system['other_specs']['beam_width']  # Optical beam divergence

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

    def get_HV57_CN(self, h, A=1.7e-14, h_0=0, v=21):
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

    def get_scintillation_index(self):   # what do to with this index???
        wave_number = 2 * pi / self.Lambda
        prop_distance = self.h / np.sin(self.elevation_angle)
        h = np.linspace(0, prop_distance, 500)
        cn_integral = np.trapz([self.get_HV57_CN(hi)*hi**(5/6) for hi in h], h)
        scintillation_index = 2.25 * wave_number **(7/6) * cn_integral
        
        return scintillation_index

    def get_Strehl_ratio_loss(self): # TODO check D_R
        """
        Strehl ratio loss
        """

        S_BS = (1 + (self.D_R/self.get_fried_parameter())**(5/3))**-6/5
        Strehl_loss = 10 * log10(S_BS)
        return Strehl_loss

    def get_fried_parameter(self):
        """
        Fried parameter
        """
        k = 2 * pi / self.Lambda
        prop_distance = self.h / np.sin(self.elevation_angle)
        h = np.linspace(0, prop_distance, 500)
        cn_integral = np.trapz([self.get_HV57_CN(hi) for hi in h], h)
        r_0 = (0.423 * k**2 * cn_integral)**(-3/5)
        return r_0

    def get_wavefront_error_loss(self):   #what is D
        """
        Wavefront error loss
        """
        S = exp(-1.03(self.D_R/self.get_fried_parameter())**(5/3))
        return 10 * log10(S)

    def get_WFE_beam_spread_loss(self):   #what is D
        """
        WFE beam spread loss
        """
        S = (1 + (self.D_R/self.get_fried_parameter())**(5/3))**(-5/6)
        return 10 * log10(S)

    def get_beam_wander_loss(self):
        """
        Beam wander loss
        """

        # beam wander loss assumed negligible for downlink
        return None

    def get_AoA_fluctuation_loss(self): # what is D???
        """
        AoA fluctuation loss
        """
        fried_parameter = self.get_fried_parameter()
        AoA_variance = 0.2 * (self.D_R/fried_parameter)**(5/3) * (self.Lambda/fried_parameter)**2

        return 10 * log10(AoA_variance)

    def get_p_outage(self): # TODO fix this
        """
        P_outage
        """
        return 1*10**-2

    def get_scintillation_loss(self): 
        """
        Scintillation loss
        """
        scintillation_term = np.sqrt(self.get_scintillation_index())**(4/5)
        scintillation_loss = 3.3 - 5.77 * np.sqrt(-np.ln(self.get_p_outage())) * scintillation_term
        return scintillation_loss


optical_system = OS.optical_system1
link_budget = LinkBudget(optical_system)

