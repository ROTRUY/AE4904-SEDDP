import numpy as np
from scipy.special import erfc
from helpers import linear_to_dB, dB_to_linear

q = 1.602e-19      # electron charge (C)
k = 1.38e-23       # Boltzmann constant (J/K)


def simple_noise(B: float | int, T_S: float | int):
    """
    Function to determine noise.

    :param B: Channel bandwidth [Hz]
    :type B: float | int
    :param T_S: System noise temperature [K]
    :type T_S: float | int 
    """
    k = 1.38e-23  # Boltzmann constant
    return k * T_S * B


class opticalNoise():
    """
    Class to determine noise statistics and BER in an intensity-modulated
    optical communication link using a Gaussian noise approximation.

    All noise terms are referred to the optical power domain at the
    photodetector input and combined via root-sum-square (RSS).

    ---

    Parameters
    ----------
    I1 : float
        Optical power level for transmitted bit "1" [W].
        This is the average received optical power at the photodetector
        corresponding to a logical '1'.

    I0 : float
        Optical power level for transmitted bit "0" [W].
        Often assumed ≈ 0 for OOK systems but may be non-zero for
        biased transmitters.

    B : float
        Receiver equivalent noise bandwidth [Hz].
        Typically approximated as ~0.7-1.0 x data rate for NRZ systems.
        All white noise sources scale with sqrt(B).

    RIN : float
        Relative Intensity Noise spectral density [1/Hz].
        Must be given in linear units (NOT dB/Hz).

    eta_PD : float
        Photodetector responsivity [A/W].
        Typical values:
            ~0.8-1.0 A/W at 1550 nm for InGaAs PIN detectors.

    eta_RX : float
        Receiver transimpedance gain [V/A].

    eta_LD : float, optional (default = 1.0)
        Laser slope efficiency [W/A].
        Converts transmitter current noise to optical power noise.
        Only relevant if transmitter current noise is included.

    eta_FO : float, optional (default = 1.0)
        Optical channel transmission coefficient (dimensionless).
        For fiber:
            eta_FO = 10^(-Loss_dB / 10)
        For free-space:
            eta_FO = total link transmission efficiency.

    I_TX_RMS : float, optional (default = 0)
        RMS current noise of laser driver electronics [A].
        Represents additive transmitter noise.

    i_dark : float, optional (default = 0)
        Photodetector dark current [A].

    R_load : float, optional (default = 50)
        Equivalent input resistance of receiver front-end [Ohm].

    T : float, optional (default = 300)
        Receiver temperature [K].

    ---        

    Assumptions
    -----------
    - On-Off Keying (OOK)
    - Equal probability of 0 and 1 bits
    - Gaussian-distributed noise
    - Optimal decision threshold
    - White noise within bandwidth B
    """

    def __init__(self, I1, I0, B, RIN, eta_PD, eta_RX, eta_LD=1.0, eta_FO=1.0, I_TX_RMS=0, i_dark=0, R_load=50, T=300):
        self.I1 = I1
        self.I0 = I0
        self.B = B
        self.RIN = RIN
        self.eta_PD = eta_PD
        self.eta_RX = eta_RX
        self.eta_LD = eta_LD
        self.eta_FO = eta_FO
        self.I_TX_RMS = I_TX_RMS
        self.i_dark = i_dark
        self.R_load = R_load
        self.T = T

        # === Calculate transmitter noise, RIN noise, shot noise, dark current noise, and Johnson noise ===
        self.sigmaTX = self.transmitterNoise()
        self.sigmaRIN1, self.sigmaRIN0 = self.RinNoise()
        self.sigmaShot1, self.sigmaShot0 = self.shotNoise()
        self.sigmaDark = self.darkCurrentNoise()
        self.sigmaJohnson = self.johnsonNoise()

        # === Calculate total noise for '1' and '0' bits ===
        self.sigma1, self.sigma0 = self.totalNoise()
    
    def transmitterNoise(self):
        return self.eta_LD * self.eta_FO * self.I_TX_RMS
    
    def RinNoise(self):
        sigma_rin_1 = self.I1 * np.sqrt(self.RIN * self.B)
        sigma_rin_0 = self.I0 * np.sqrt(self.RIN * self.B)
        return sigma_rin_1, sigma_rin_0
    
    def shotNoise(self):
        sigma_shot_1 = np.sqrt((2 * q * self.I1 * self.B) / self.eta_PD)
        sigma_shot_0 = np.sqrt((2 * q * self.I0 * self.B) / self.eta_PD)
        return sigma_shot_1, sigma_shot_0
    
    def darkCurrentNoise(self):
        return np.sqrt(2 * q * self.i_dark * self.B) / self.eta_PD
    
    def johnsonNoise(self):
        return np.sqrt(4 * k * self.T * self.R_load * self.B) / (self.eta_PD * self.eta_RX)
    
    def totalNoise(self):
        sigma1 = np.sqrt(self.sigmaTX**2 + self.sigmaRIN1**2 + self.sigmaShot1**2 + self.sigmaDark**2 + self.sigmaJohnson**2)
        sigma0 = np.sqrt(self.sigmaTX**2 + self.sigmaRIN0**2 + self.sigmaShot0**2 + self.sigmaDark**2 + self.sigmaJohnson**2)
        return sigma1, sigma0
    
    def Q(self):
        return (self.I1 - self.I0) / (self.sigma1 + self.sigma0)
    
    def BER(self):
        Q_value = self.Q()
        return 0.5 * erfc(Q_value / np.sqrt(2))
    
    def summary(self, in_dB: bool = False, Print: bool = True, save: bool = False, filename: str = "opticalNoiseSummary.txt"):
        if in_dB:
            summary_str = f"""
            === Optical Noise Summary ===
            --- Inputs ---
            I1 (Optical Power for '1'): {self.I1:.3e} W
            I0 (Optical Power for '0'): {self.I0:.3e} W
            B (Bandwidth): {self.B:.3e} Hz
            RIN: {self.RIN:.3e} 1/Hz
            eta_PD (Responsivity): {self.eta_PD:.3f} A/W
            eta_RX (Transimpedance Gain): {self.eta_RX:.3f} V/A
            eta_LD (Laser Slope Efficiency): {self.eta_LD:.3f} W/A
            eta_FO (Channel Transmission): {self.eta_FO:.3f}
            I_TX_RMS (Transmitter Current Noise): {self.I_TX_RMS:.3e} A
            i_dark (Dark Current): {self.i_dark:.3e} A
            R_load (Load Resistance): {self.R_load:.3f} Ohm
            T (Temperature): {self.T:.1f} K

            --- Noise Components ---
            Transmitter Noise: {linear_to_dB(self.sigmaTX):.3f} dBW
            RIN Noise for '1': {linear_to_dB(self.sigmaRIN1):.3f} dBW
            RIN Noise for '0': {linear_to_dB(self.sigmaRIN0):.3f} dBW
            Shot Noise for '1': {linear_to_dB(self.sigmaShot1):.3f} dBW
            Shot Noise for '0': {linear_to_dB(self.sigmaShot0):.3f} dBW
            Dark Current Noise: {linear_to_dB(self.sigmaDark):.3f} dBW
            Johnson Noise: {linear_to_dB(self.sigmaJohnson):.3f} dBW

            --- Total Noise ---
            Total Noise for '1': {linear_to_dB(self.sigma1):.3f} dBW
            Total Noise for '0': {linear_to_dB(self.sigma0):.3f} dBW

            --- Performance Metrics ---
            Q-factor: {self.Q():.2f}
            BER: {self.BER():.2e}
            """
        else:
            summary_str = f"""
            === Optical Noise Summary ===
            --- Inputs ---
            I1 (Optical Power for '1'): {self.I1:.3e} W
            I0 (Optical Power for '0'): {self.I0:.3e} W
            B (Bandwidth): {self.B:.3e} Hz
            RIN: {self.RIN:.3e} 1/Hz
            eta_PD (Responsivity): {self.eta_PD:.3f} A/W
            eta_RX (Transimpedance Gain): {self.eta_RX:.3f} V/A
            eta_LD (Laser Slope Efficiency): {self.eta_LD:.3f} W/A
            eta_FO (Channel Transmission): {self.eta_FO:.3f}
            I_TX_RMS (Transmitter Current Noise): {self.I_TX_RMS:.3e} A
            i_dark (Dark Current): {self.i_dark:.3e} A
            R_load (Load Resistance): {self.R_load:.3f} Ohm
            T (Temperature): {self.T:.1f} K

            --- Noise Components ---
            Transmitter Noise: {self.sigmaTX:.3e} W
            RIN Noise for '1': {self.sigmaRIN1:.3e} W
            RIN Noise for '0': {self.sigmaRIN0:.3e} W
            Shot Noise for '1': {self.sigmaShot1:.3e} W
            Shot Noise for '0': {self.sigmaShot0:.3e} W
            Dark Current Noise: {self.sigmaDark:.3e} W
            Johnson Noise: {self.sigmaJohnson:.3e} W

            --- Total Noise ---
            Total Noise for '1': {self.sigma1:.3e} W
            Total Noise for '0': {self.sigma0:.3e} W

            --- Performance Metrics ---
            Q-factor: {self.Q():.2f}
            BER: {self.BER():.2e}
            """
        if Print:
            print(summary_str)
        if save:
            with open(filename, 'w') as f:
                f.write(summary_str)

### RUN HERE
example = opticalNoise(I1=1e-3, I0=1e-6, B=10e9, RIN=dB_to_linear(-145), eta_PD=0.9, eta_RX=1000, eta_LD=0.5, eta_FO=1, I_TX_RMS=1e-6, i_dark=10e-9, R_load=50, T=300)
example.summary(True)