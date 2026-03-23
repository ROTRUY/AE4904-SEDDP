# IMPORTS
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
        self.elevation_angle = optical_system['link_benchmark_specs']['elevation_angle']
        self.D_T = optical_system['transmitter_specs']['transmitter_aperture']  # Transmitter aperture
        self.D_R = optical_system['receiver_specs']['receiver_aperture']  # Receiver aperture
        self.Lambda = optical_system['transmitter_specs']['wavelength']  # Wavelength
        c = 3e+8  # Speed of light
        self.freq = c / optical_system['transmitter_specs']['wavelength']  # Frequency from wavelength
        self.h = optical_system['link_benchmark_specs']['altitude']  # altitude [m]
        self.R = self.h / np.sin(self.elevation_angle)  # Link range
        self.receiver_outage_power = optical_system['receiver_specs']['receiver_outage_power']
        self.outage_probability = optical_system['receiver_specs'].get('outage_probability', 1e-3)
        self.receiver_threshold_dbm = optical_system['receiver_specs']['receiver_threshold_dbm']
        self.transmission_optics = optical_system['transmitter_specs']['transmission_optics']
        self.fsm_bandwidth = optical_system['transmitter_specs']['fsm_bandwidth']   # Hz
        self.psd_amplitude = optical_system['transmitter_specs']['psd_amplitude']    # rad2/Hz
        self.psd_corner_freq = optical_system['transmitter_specs']['psd_corner_freq']  # Hz

        self.transmitter_divergence_angle = self.get_transmitter_divergence_angle()
        self.transmitter_pointing_error = self.get_pointing_jitter() # last correction is fsm

    def get_transmitter_pointing_error(self):
        return self.get_pointing_jitter()

    def get_transmitter_divergence_angle(self):
        """
        Divergence angle of transmitter
        """
        theta_div = self.Lambda / self.D_T
        return theta_div

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
        """
        A = self.psd_amplitude    # rad^2/Hz
        f_c = self.psd_corner_freq  # Hz

        # Variance of one axis
        sigma_ol = A * f_c * pi/2
        return np.sqrt(sigma_ol)  # 1-axis RMS [rad]

    def get_pointing_jitter(self):
        """
        Post-FSM residual jitter: integral from fsm_bandwidth to infinity.
        FSM acts as a high-pass filter; only energy above f_BW remains.

        For Lorentzian S(f) = A / (1 + (f/f_c)^2):
            ∫_{f_BW}^{inf} S(f) df = A * f_c * (pi/2 - arctan(f_BW / f_c))
        """
        A = self.psd_amplitude
        f_c = self.psd_corner_freq

        sigma_sq = A * f_c * (np.pi / 2 - np.arctan(self.fsm_bandwidth / f_c))
        return np.sqrt(sigma_sq)
        # return self.fsm_accuracy  # 1e-6 rad

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
        cn_integral = np.trapezoid([self.get_HV57_CN(hi)*hi**(5/6) for hi in h], h)
        scintillation_index = 2.25 * wave_number ** (7/6) * cn_integral
        return scintillation_index

    def get_Strehl_ratio_loss(self):  # TODO check D_R
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
        cn_integral = np.trapezoid([self.get_HV57_CN(hi) for hi in h], h)
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

    def get_AoA_fluctuation_loss(self):  # what is D???
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

    def get_link_margin(self, laser_power):
        """
        Link margin relative to receiver threshold [dB].
        """
        return self.get_total_link_budget(laser_power) + self.receiver_threshold_dbm


optical_system = OS.optical_system1
link_budget = LinkBudget(optical_system)

power_dbm = optical_system["transmitter_specs"]["transmitter_laser_power"]
total_budget_dbm = link_budget.get_total_link_budget(power_dbm)
receiver_threshold_dbm = optical_system["receiver_specs"]["receiver_threshold_dbm"]
link_margin_db = link_budget.get_link_margin(power_dbm)

label_width = 38

print("Computing Link Budget...\n")
print("Method: Transmitter + Atmospheric + Receiver\n")

print("Transmitter:")
print(f"  {('Transmitter power'): <{label_width}} {power_dbm:>8.2f} dBm")
print(f"  {('Transmitter gain'): <{label_width}} {link_budget.get_transmitter_gain():>8.2f} dB")
print(f"  {('Free space loss'): <{label_width}} {link_budget.get_free_space_loss():>8.2f} dB")
print(f"  {('Transmission loss'): <{label_width}} {link_budget.get_transmission_loss():>8.2f} dB")
print(f"  {('Atmospheric loss'): <{label_width}} {link_budget.get_atmospheric_loss():>8.2f} dB")

print("\nPointing losses:")
print(f'- Static pointing error loss: {link_budget.get_static_pointing_error_loss():.2f} dB')
print(f'- Average pointing jitter loss: {link_budget.get_avg_pointing_jitter_loss():.2f} dB \n')
print(f'- Pointing jitter induced scintillation loss: {link_budget.get_pointing_jitter_scintillation_loss():.2f} dB')
print(f'- Sigma_pj: {1e6*link_budget.get_pointing_jitter():.2f} µrad')
print(f'- Sigma_ol: {1e6*link_budget.get_pointing_jitter_openloop():.2f} µrad')

print("\nAtmospheric losses:")
print(
    f"  {('WFE and beam spread losses'): <{label_width}} {link_budget.get_WFE_beam_spread_loss():>8.2f} dB"
)
print(
    f"  {('Beam wander losses'): <{label_width}} {link_budget.get_beam_wander_loss():>8.2f} dB"
)
print(
    f"  {('Scintillation losses'): <{label_width}} {link_budget.get_scintillation_loss():>8.2f} dB"
)

print("\nOther:")
print(f"  {('Scintillation index'): <{label_width}} {link_budget.get_scintillation_index():>8.2f}")
print(f"  {('Fried parameter'): <{label_width}} {link_budget.get_fried_parameter():>8.2f}")

print("\nReceiver:")
print(f"  {('Gain receiver'): <{label_width}} {link_budget.get_receiver_gain():>8.2f} dB")
print(f"  {('Receiver losses'): <{label_width}} {link_budget.get_receiver_losses():>8.2f} dB")

print(f"\nTotal losses (sum): {total_budget_dbm:.2f}")
print(f"Link Budget: {link_budget.get_total_link_budget(26)}")
print(f"Receiver threshold: {receiver_threshold_dbm:.2f} dBm")
print(f"Link margin: {link_margin_db:.2f} dB")
