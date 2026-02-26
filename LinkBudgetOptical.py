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
        self.transmitter_divergence_angle = optical_system['transmitter_specs']['transmitter_divergence_angle']
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

    def get_transmitter_gain(self):
        """
        Gain of transmitter
        """
        gain_TX = 8 / (self.transmitter_divergence_angle)**2
        return gain_TX

    def get_receiver_gain(self):
        """
        Gain of receiver
        """
        gain_RX = (np.pi * self.D_R / (self.Lambda))**2
        return gain_RX

# TODO define pointing error, pointing jitter RMS, outage probability

    def get_pointing_jitter(self):
        """
        Platform jitter
        """
        def get_PSD(f):
            return 160 / (1 + f**2) 

        # f = np.linspace(0, 1000, 1000)
        # pointing_jitter_variance = np.trapz([get_PSD(f) for fi in f], f)
        pointing_jitter = 160 * np.arctan(1000) * 10^-6 # radians (sigma_pj^2)
        return pointing_jitter


    def get_static_pointing_error_loss(self): # what is pointing error?
        """
        Static pointing error loss
        """
        T_pe = exp(-2 * self.transmitter_static_pointing_error**2 / self.transmitter_divergence_angle**2)
        return 10 * log10(T_pe)

    def get_avg_pointing_jitter_loss(self):
        """
        Average pointing jitter loss
        """
        pointing_jitter = self.get_pointing_jitter()
        T_pa = self.transmitter_divergence_angle**2 / (self.transmitter_divergence_angle**2 + 4 * pointing_jitter**2) 
        return 10 * log10(T_pa)

    def get_pointing_jitter_scintillation_loss(self):
        """
        Pointing jitter scintillation loss
        """
        pointing_jitter = self.get_pointing_jitter()
        outage_probability = self.()
        T_ps = outage_probability**(4*pointing_jitter**2 / self.transmitter_divergence_angle**2)
        return 10 * log10(T_ps)




    def get_pointing_jitter_RMS(self): # what is pointing jitter RMS?
        """
        Pointing jitter RMS
        """

        return self.theta_div / np.sqrt(12)


    
    def L_FS(self):
        """
        Free-space propagation loss
        """
        return (4 * pi * self.R / self.Lambda)**2


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

print('Computing Link Budget...\n')

print('Method: Transmitter + Medium + Receiver\n')

print('Transmitter:\n')

print("Pointing losses:")
print(f'- Static pointing error loss: {link_budget.get_static_pointing_error_loss()} dB')
print(f'- Average pointing jitter loss: {link_budget.get_avg_pointing_jitter_loss()} dB')
print(f'- Pointing jitter induced scintillation loss: {link_budget.get_pointing_jitter_scintillation_loss()} dB')



print(f'Transmitter power: {optical_system['transmitter_specs']['transmitter_laser_power']} dBm')
print(f'Transmitter gain: {link_budget.get_transmitter_gain()} dB')
print(f'Transmitter losses: {}')

print('Medium:\n')

print('Receiver:\n')

print(f'Gain receiver: {link_budget.get_receiver_gain()} dB')
print(f'Receiver losses: {}')

print('Final Link Budget:\n')
