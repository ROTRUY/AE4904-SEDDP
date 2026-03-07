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


import numpy as np


class OpticalNoiseV2:

    def __init__(self, P_R:float, wavelength:float, B_e:float, T:float=300, R:float=50, eta:float=0.9, i_D:float=1e-9, detector_type:str="PIN", detector_material:str="ingaas", M:float=1):
        """
        :param P_R: Received optical power at the photodetector [W]
        :type P_R: float
        :param wavelength: Wavelength of the optical signal [m]
        :type wavelength: float
        :param B_e: Electrical bandwidth of the receiver [Hz]
        :type B_e: float
        :param T: Receiver temperature [K], default is 300 K
        :type T: float
        :param R: Load resistance [Ohm], default is 50 Ohm
        :type R: float
        :param eta: Photodiode quantum efficiency (0.30 to 0.95 typical), default is 0.9
        :type eta: float
        :param i_D: Dark current of the photodiode [A], default is 1 nA
        :type i_D: float
        :param detector_type: Type of photodetector ("PIN" or "APD"), default is "PIN"
        :type detector_type: str
        :param detector_material: Material of the photodetector ("Si", "InGaAs", "Ge"), default is "InGaAs"
        :type detector_material: str
        :param M: Avalanche gain for APD (ignored for PIN), default is 1
        :type M: float
        """
        self.P_R = P_R  # Received optical power [W]
        self.wavelength = wavelength  # Wavelength [m]
        self.B_e = B_e  # Electrical receiver bandwidth [Hz]
        self.T = T  # Receiver temperature [K]
        self.R = R  # Load resistance [Ohm]
        self.eta = eta  # Photodiode quantum efficiency [-] (30% to 95% typical)
        self.i_D = i_D  # Dark current [A]

        self.detector_type = detector_type.upper()  # "PIN" or "APD"
        self.detector_material = detector_material.lower()  # "si", "ingaas" or "ge"

        match self.detector_type:
            case "APD":
                match self.detector_material:
                    case "si":
                        x = 0.3
                    case "ingaas":
                        x = 0.7
                    case "ge":
                        x = 1.0
                    case _:
                        raise ValueError("Invalid detector material. Must be 'Si', 'InGaAs', or 'Ge'.")
                self.M = M  # Avalanche gain for APD
                self.F = M ** x  # Excess noise factor for APD
            case "PIN":
                self.M = 1  # No gain for PIN
                self.F = 1  # No excess noise for PIN
            case _:
                raise ValueError("Invalid detector type. Must be 'PIN' or 'APD'.")

        # constants
        self.q = 1.602176634e-19  # electron charge (C)
        self.h = 6.62607015e-34  # Planck's constant (J*s)
        self.c = 299792458  # speed of light (m/s)
        self.kB = 1.380649e-23  # Boltzmann constant (J/K)

        # Calculate everything
        self.sigmaShot = self.shotNoise()
        self.sigmaDark = self.darkNoise()
        self.sigmaThermal = self.thermalNoise()
        self.sigmaTotal = self.totalNoise()
        self.SNR = self.SNR()

    def responsivity(self):
        """
        Calculate photodetector responsivity [A/W] based on quantum efficiency and wavelength.
        """
        return self.eta * self.q * self.wavelength / (self.h * self.c) * self.M

    def photocurrent(self):
        """
        Calculate photocurrent generated by the received optical power [A].
        """
        return self.responsivity() * self.P_R

    def shotNoise(self):
        """
        Calculate shot noise. [A^2]
        """
        return 2 * self.q * self.photocurrent() * self.B_e * (self.M ** 2) * self.F

    def darkNoise(self):
        """
        Calculate dark current noise. [A^2]
        """
        return 2 * self.q * self.i_D * self.B_e * (self.M ** 2) * self.F

    def thermalNoise(self):
        """
        Calculate thermal noise. [A^2]
        """
        return 4 * self.kB * self.T / self.R * self.B_e

    def totalNoise(self):
        """
        Calculate total noise. [A^2]
        """
        return self.sigmaShot + self.sigmaDark + self.sigmaThermal

    def SNR(self, ):
        """
        Calculate signal-to-noise ratio (SNR) at the photodetector output (dimensionless).
        """
        return self.photocurrent() ** 2 / self.totalNoise()

    def summary(self, in_dB: bool = False, Print: bool = True, save: bool = False, filename: str = "opticalNoiseSummary.txt"):
        if in_dB:
            summary_str = f"""
            === Optical Noise Summary ===
            --- Inputs ---
            P_R (Received Optical Power): {self.P_R:.3e} W
            Wavelength: {self.wavelength:.3e} m
            B_e (Electrical Bandwidth): {self.B_e:.3e} Hz
            T (Temperature): {self.T:.1f} K
            R (Load Resistance): {self.R:.3f} Ohm
            eta (Quantum Efficiency): {self.eta:.3f}
            i_D (Dark Current): {self.i_D:.3e} A
            Detector Type: {self.detector_type}
            Detector Material: {self.detector_material}
            Avalanche Gain (M): {self.M:.3f}

            --- Noise Components ---
            Shot noise: {linear_to_dB(self.sigmaShot):.3f} dBA
            Dark current noise: {linear_to_dB(self.sigmaDark):.3f} dBA
            Thermal noise: {linear_to_dB(self.sigmaThermal):.3f} dBA

            --- Total Noise ---
            Total noise: {linear_to_dB(self.sigmaTotal):.3f} dBA

            --- Performance Metrics ---
            SNR: {linear_to_dB(self.SNR):.3f} dB
            """
        else:
            summary_str = f"""
            === Optical Noise Summary ===
            --- Inputs ---
            P_R (Received Optical Power): {self.P_R:.3e} W
            Wavelength: {self.wavelength:.3e} m
            B_e (Electrical Bandwidth): {self.B_e:.3e} Hz
            T (Temperature): {self.T:.1f} K
            R (Load Resistance): {self.R:.3f} Ohm
            eta (Quantum Efficiency): {self.eta:.3f}
            i_D (Dark Current): {self.i_D:.3e} A
            Detector Type: {self.detector_type}
            Detector Material: {self.detector_material}
            Avalanche Gain (M): {self.M:.3f}

            --- Noise Components ---
            Shot noise: {self.sigmaShot:.3e} A^2
            Dark current noise: {self.sigmaDark:.3e} A^2
            Thermal noise: {self.sigmaThermal:.3e} A^2

            --- Total Noise ---
            Total noise: {self.sigmaTotal:.3e} A^2

            --- Performance Metrics ---
            SNR: {self.SNR:.3e}
            """
        if Print:
            print(summary_str)
        if save:
            with open(filename, 'w') as f:
                f.write(summary_str)


### RUN HERE
#example = opticalNoise(I1=1e-3, I0=1e-6, B=10e9, RIN=dB_to_linear(-145), eta_PD=0.9, eta_RX=1000, eta_LD=0.5, eta_FO=1, I_TX_RMS=1e-6, i_dark=10e-9, R_load=50, T=300)
#example.summary(True)

exampleV2 = OpticalNoiseV2(P_R=1e-3, wavelength=1550e-9, B_e=10e9, T=300, R=50, eta=0.9, i_D=1e-9, detector_type="APD", detector_material="InGaAs", M=10)
exampleV2.summary(True)