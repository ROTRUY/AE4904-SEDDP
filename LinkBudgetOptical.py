### IMPORTS
from math import pi, exp, log10, sqrt
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
        self.elevation_angle = optical_system['link_benchmark_specs']['elevation_angle']
        self.transmitter_divergence_angle = optical_system['transmitter_specs']['transmitter_divergence_angle']
        self.D_T = optical_system['transmitter_specs']['transmitter_aperture']  # Transmitter aperture
        self.D_R = optical_system['receiver_specs']['receiver_aperture']  # Receiver aperture
        self.Lambda = optical_system['transmitter_specs']['system_frequency']  # Wavelength
        c = 3e+8  # Speed of light
        self.freq = c / optical_system['transmitter_specs']['system_frequency']  # Frequency from wavelength
        self.h = optical_system['link_benchmark_specs']['altitude']  # altitude [m]
        self.R = self.h / np.sin(self.elevation_angle)  # Link range
        self.theta = optical_system['transmitter_specs']['platform_drift_angle']  # Beam jitter angle
        self.transmitter_pointing_error = optical_system['transmitter_specs']['transmitter_pointing_error']
        self.receiver_outage_power = optical_system['receiver_specs']['receiver_outage_power']
        self.outage_probability = optical_system['receiver_specs'].get('outage_probability', 1e-3)
        self.transmission_optics = optical_system['transmitter_specs']['transmission_optics']
        self.fsm_bandwidth = optical_system['transmitter_specs']['fsm_bandwidth']   # Hz
        self.fsm_accuracy = optical_system['transmitter_specs']['fsm_accuracy']  # rad
        self.psd_amplitude = optical_system['transmitter_specs']['psd_amplitude']    # rad2/Hz
        self.psd_corner_freq = optical_system['transmitter_specs']['psd_corner_freq']  # Hz

    def get_transmitter_gain(self):
        """
        Gain of transmitter
        """
        gain_TX = 8 / (self.transmitter_divergence_angle)**2
        return 10 * log10(gain_TX)

    def get_receiver_gain(self):
        """
        Gain of receiver
        """
        gain_RX = (np.pi * self.D_R / (self.Lambda))**2
        return 10 * log10(gain_RX)

    def get_receiver_losses(self):
        """
        Losses of receiver
        """
        receiver_transmission = 0.5
        return 10 * log10(receiver_transmission)

    def get_free_space_loss(self):
        """
        Free space loss
        """
        fsl = (4 * np.pi * self.R / self.Lambda)**2
        return - 10 * log10(fsl)

    def get_transmission_loss(self):
        """
        Transmission loss
        """
        return 10 * log10(self.transmission_optics)

    def get_atmospheric_loss(self):
        """
        First, compute Kruse model exponent q based on visibility.
        """
        visibility_km = 10  # visibility as provided by meteorological data TODO change
        wavelength = self.Lambda 
        atmosphere_height_km = 4  # reference height for Kruse model
        zenith_angle = pi/2 - self.elevation_angle  # zenith angle

        if visibility_km > 50:
            q = 1.6
        elif 6 < visibility_km <= 50:
            q = 1.3
        elif 1 < visibility_km <= 6:
            q = 0.16 * visibility_km + 0.34
        else:
            q = 0.0  # heavy fog regime
        
        """
        Second, compute atmospheric attenuation in dB
        using Beer–Lambert + Kruse model.
        """

        alpha_lambda = (3.912 / visibility_km) * (wavelength / (550*1e-9)) ** (-q)

        L_km = atmosphere_height_km / np.cos(zenith_angle)

        loss = 4.343 * alpha_lambda * L_km

        return - loss

    def get_pointing_jitter_openloop(self):
        """
        Open-loop platform pointing jitter RMS [rad], 1-axis.
        Raw platform vibration before FSM correction.

        Model: Lorentzian PSD  S(f) = A / (1 + (f/f_c)^2)  [rad^2/Hz]
        Variance: sigma^2 = integral_0^{f_max} S(f) df
                          = A * f_c * arctan(f_max / f_c)
        - Single-axis jitter; total 2D radial sigma = sqrt(2) * sigma_1axis
        (isotropic Gaussian assumed, consistent with pointing_jitter.py)
        - Lorentzian PSD is a first-order approximation; no resonance peaks
        - Integration up to fsm_bandwidth only
        """
        A = self.psd_amplitude    # rad^2/Hz
        f_c = self.psd_corner_freq  # Hz
        f_max = self.fsm_bandwidth  # Hz

        # Variance of one axis
        sigma_pj = A * f_c * np.arctan(f_max / f_c)  # integral of S(f)
        return np.sqrt(sigma_pj)  # 1-axis RMS [rad]
    
    def get_pointing_jitter(self):
        """
        Post-FSM residual pointing jitter RMS [rad], 1-axis.
        FSM suppresses open-loop platform jitter; residual = fsm_accuracy.
        """
        return self.fsm_accuracy  # 1e-6 rad

    def get_static_pointing_error_loss(self):
        """
        Static pointing error loss
        """
        T_pe = exp(-2 * self.transmitter_pointing_error**2 / self.transmitter_divergence_angle**2)
        return 10 * log10(T_pe)

    def get_avg_pointing_jitter_loss(self):
        """
        Average pointing jitter loss [dB].
        """
        sigma_pj = self.get_pointing_jitter()
        T_pa = (self.transmitter_divergence_angle**2 / (self.transmitter_divergence_angle**2 + 4 * sigma_pj**2))
        return 10 * log10(T_pa)

    def get_pointing_jitter_scintillation_loss(self):
        """
        Pointing jitter scintillation loss
        """
        sigma_pj = self.get_pointing_jitter()
        P0 = self.outage_probability
        exponent = 4 * sigma_pj**2 / self.transmitter_divergence_angle**2
        T_ps = P0 ** exponent
        return 10 * log10(T_ps)

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

        S_BS = (1 + (self.D_R/self.get_fried_parameter())**(5/3))**(-6/5)
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
        S = exp(-1.03 * (self.D_R/self.get_fried_parameter())**(5/3))
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
        return 0

    def get_AoA_fluctuation_loss(self): # what is D???
        """
        AoA fluctuation loss
        """
        fried_parameter = self.get_fried_parameter()
        AoA_variance = 0.2 * (self.D_R/fried_parameter)**(5/3) * (self.Lambda/fried_parameter)**2

        return 10 * log10(AoA_variance)

    def get_p_outage(self):
        """Outage probability; from config."""
        return self.outage_probability

    def get_scintillation_loss(self): 
        """
        Scintillation loss
        """
        sigma_I = np.sqrt(self.get_scintillation_index())
        scintillation_loss = (3.3 - 5.77 * np.sqrt(-np.log(self.get_p_outage()))) * sigma_I**(4/5)
        return scintillation_loss
    
    def get_total_link_budget(self, laser_power):
        """
        Total link budget
        """
        total = laser_power
        total += self.get_transmitter_gain()
        total += self.get_free_space_loss()
        total += self.get_transmission_loss()
        total += self.get_atmospheric_loss()
        total += self.get_static_pointing_error_loss()
        total += self.get_avg_pointing_jitter_loss()
        total += self.get_WFE_beam_spread_loss()
        total += self.get_beam_wander_loss()
        total += self.get_scintillation_loss()
        total += self.get_receiver_gain()
        total += self.get_receiver_losses()

        return total


optical_system = OS.optical_system1
link_budget = LinkBudget(optical_system)

print('Computing Link Budget...\n')

print('Method: Transmitter + Atmospheric + Receiver\n')

print('Transmitter:\n') 

print(f'- Transmitter power: {optical_system["transmitter_specs"]["transmitter_laser_power"]:.2f} dBm')
print(f'- Transmitter gain: {link_budget.get_transmitter_gain():.2f} dB')
print(f'- Free space loss: {link_budget.get_free_space_loss():.2f} dB')
print(f'- Transmission loss: {link_budget.get_transmission_loss():.2f} dB')
print(f'- Atmospheric loss: {link_budget.get_atmospheric_loss():.2f} dB')


print("\nPointing losses:")
print(f'- Static pointing error loss: {link_budget.get_static_pointing_error_loss():.2f} dB')
print(f'- Average pointing jitter loss: {link_budget.get_avg_pointing_jitter_loss():.2f} dB \n')
print(f'- Pointing jitter induced scintillation loss: {link_budget.get_pointing_jitter_scintillation_loss():.2f} dB')
# print(f'- Sigma_pj: {link_budget.get_pointing_jitter()}')

print('Atmospheric losses:\n')

print(f'- WFE and beam spread losses: {link_budget.get_WFE_beam_spread_loss():.2f} dB')
print(f'- Beam wander losses: {link_budget.get_beam_wander_loss():.2f} dB')
print(f'- Scintillation losses: {link_budget.get_scintillation_loss():.2f} dB')

print("\nOther \n")
print(f'scintillation index: {link_budget.get_scintillation_index():.2f}')
print(f'fried parameter: {link_budget.get_fried_parameter():.2f}')

print('\nReceiver:\n')

print(f'Gain receiver: {link_budget.get_receiver_gain():.2f} dB')
print(f'Receiver losses: {link_budget.get_receiver_losses():.2f} dB')

print(f'\n\nTotal losses (sum): {link_budget.get_total_link_budget(optical_system["transmitter_specs"]["transmitter_laser_power"]):.2f}')

print(f'Receiver threshold: 30 dBm')

print(f'Link margin: {link_budget.get_total_link_budget(optical_system["transmitter_specs"]["transmitter_laser_power"]) + 30:.2f} dB')