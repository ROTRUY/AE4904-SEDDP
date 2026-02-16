import math

# Constants

c = 299792458.0
# Speed of light [m/s]

k_db = -228.6
# 10*log10(Boltzmann constant) [dBW/K/Hz]

pi = math.pi
# Pi


# Helper functions

def db_to_lin(x_db):
    """Convert value from dB to linear (power ratio)."""
    return 10 ** (x_db / 10.0)

def lin_to_db(x_lin):
    """Convert value from linear (power ratio) to dB."""
    return 10.0 * math.log10(x_lin)

def w_to_dbw(P_w):
    """Convert power from watts [W] to dBW."""
    if P_w <= 0:
        raise ValueError("P_tx_W must be > 0")
    return 10.0 * math.log10(P_w)


# Link budget functions

def wavelength_m(f_hz):
    """Wavelength [m] from carrier frequency f [Hz]."""
    if f_hz <= 0:
        raise ValueError("f_hz must be > 0")
    return c / f_hz

def fspl_db(R_m, f_hz):
    """Free-space path loss L_fs [dB] from range R [m] and frequency f [Hz]."""
    if R_m <= 0:
        raise ValueError("R_m must be > 0")
    lam = wavelength_m(f_hz)
    return 20.0 * math.log10(4.0 * pi * R_m / lam)

def eirp_dbw(P_tx_dBW, G_tx_dBi, L_tx_dB, L_point_tx_dB=0.0):
    """Equivalent isotropically radiated power EIRP [dBW]."""
    return P_tx_dBW + G_tx_dBi - L_tx_dB - L_point_tx_dB

def total_path_loss_db(L_fs_dB, L_atm_dB=0.0, L_rain_dB=0.0, L_pol_dB=0.0, L_point_rx_dB=0.0, L_misc_dB=0.0):
    """Total propagation/path loss L_path [dB]."""
    return L_fs_dB + L_atm_dB + L_rain_dB + L_pol_dB + L_point_rx_dB + L_misc_dB

def received_carrier_power_dbw(EIRP_dBW, L_path_dB, G_rx_dBi):
    """Received carrier power C [dBW] at antenna terminals."""
    return EIRP_dBW - L_path_dB + G_rx_dBi

def noise_density_dbw_per_hz(T_sys_K):
    """Noise power spectral density N0 [dBW/Hz] from system noise temperature T_sys [K]."""
    if T_sys_K <= 0:
        raise ValueError("T_sys_K must be > 0")
    return k_db + 10.0 * math.log10(T_sys_K)

def cn0_dbhz(C_dBW, N0_dBW_per_Hz):
    """Carrier-to-noise-density ratio C/N0 [dB-Hz]."""
    return C_dBW - N0_dBW_per_Hz

def ebn0_db(CN0_dBHz, Rb_bps, L_impl_dB=0.0):
    """Available Eb/N0 [dB] from C/N0 [dB-Hz] and bit rate Rb [bit/s]."""
    if Rb_bps <= 0:
        raise ValueError("Rb_bps must be > 0")
    Rb_dB = 10.0 * math.log10(Rb_bps)
    return CN0_dBHz - Rb_dB - L_impl_dB


# Throughput functions

def symbol_rate_from_bandwidth(B_occ_hz, alpha):
    """
    Symbol rate Rs [sym/s] from occupied bandwidth B_occ [Hz].

    Assumption: raised-cosine / RRC shaping, null-to-null occupied bandwidth:
        B_occ ≈ (1 + alpha) * Rs
    """
    if B_occ_hz <= 0:
        raise ValueError("B_occ_hz must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    return B_occ_hz / (1.0 + alpha)

def gross_bit_rate_from_symbol_rate(Rs_sps, M):
    """Gross PHY bit rate Rb [bit/s] from symbol rate Rs [sym/s] and constellation size M."""
    if Rs_sps <= 0:
        raise ValueError("Rs_sps must be > 0")
    if M <= 1:
        raise ValueError("M must be > 1")
    bits_per_symbol = math.log2(M)
    return Rs_sps * bits_per_symbol


# BER functions (QPSK uncoded coherent Gray, AWGN)

def ber_qpsk_uncoded_awgn_from_ebn0_lin(ebn0_lin):
    """Uncoded coherent Gray QPSK BER in AWGN from Eb/N0 (linear)."""
    if ebn0_lin < 0:
        raise ValueError("EbN0_lin must be >= 0")
    # BER = 0.5 * erfc( sqrt(Eb/N0) )
    return 0.5 * math.erfc(math.sqrt(ebn0_lin))

def required_ebn0_for_target_ber(BER_target, tol=1e-12, max_iter=200):
    """
    Required Eb/N0 (linear and dB) to achieve target BER for uncoded QPSK in AWGN.

    Solve Eb/N0 such that:
        BER_target = 0.5 * erfc( sqrt(Eb/N0) )
    """
    if not (0.0 < BER_target < 0.5):
        raise ValueError("BER_target must be between 0 and 0.5 (exclusive)")

    lo = 0.0
    hi = 1.0

    while ber_qpsk_uncoded_awgn_from_ebn0_lin(hi) > BER_target:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("Failed to bracket solution for Eb/N0. Check BER_target.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ber_mid = ber_qpsk_uncoded_awgn_from_ebn0_lin(mid)

        if ber_mid > BER_target:
            lo = mid
        else:
            hi = mid

        if (hi - lo) / max(1.0, hi) < tol:
            break

    ebn0_req_lin = hi
    ebn0_req_dB = lin_to_db(hi)
    return ebn0_req_lin, ebn0_req_dB

def rb_max_for_target(CN0_dBHz, EbN0_req_dB, L_impl_dB=0.0):
    """Maximum gross PHY bit rate Rb_max [bit/s] that meets target Eb/N0 at given C/N0."""
    # 10log10(Rb_max) = CN0 - EbN0_req - L_impl
    rb_max_db = CN0_dBHz - EbN0_req_dB - L_impl_dB
    return 10 ** (rb_max_db / 10.0)

# ============================================================
# USER INPUTS (edit these)
# ============================================================

# Geometry / frequency
f_hz = 20e9
# Carrier frequency [Hz]

R_m = 1200e3
# Slant range between spacecraft and ground station [m]

# Transmit chain
P_tx_dBW = 10.0
# Transmit RF power at HPA output [dBW]

G_tx_dBi = 30.0
# Transmit antenna gain [dBi]

L_tx_dB = 1.0
# Transmit-side losses [dB]

L_point_tx_dB = 0.0
# Transmit pointing loss [dB]

# Propagation losses (benchmark)
L_atm_dB = 0.5
# Atmospheric/gaseous attenuation [dB]

L_rain_dB = 0.0
# Rain attenuation [dB]

L_pol_dB = 0.0
# Polarization mismatch loss [dB]

L_point_rx_dB = 0.0
# Receive pointing loss [dB]

L_misc_dB = 0.0
# Miscellaneous additional losses/margins in path [dB]

# Receiver
G_rx_dBi = 55.0
# Receive antenna gain [dBi]

T_sys_K = 500.0
# System noise temperature at antenna terminals [K]


B_occ_hz = 100e6
# Occupied (null-to-null) bandwidth used in design [Hz]

alpha = 0.2
# Roll-off factor for raised cosine / RRC pulse shaping [-]

M = 4
# Constellation size (QPSK) [-]

# BER / margin
BER_target = 1e-6
# Target bit error rate [-]

L_impl_dB = 0.0
# Implementation loss applied at Eb/N0 level [dB]

# ============================================================
# CALCULATIONS
# ============================================================

# Throughput (gross PHY)
Rs_sps = symbol_rate_from_bandwidth(B_occ_hz, alpha)
# Symbol rate [sym/s]

Rb_gross_bps = gross_bit_rate_from_symbol_rate(Rs_sps, M)
# Gross PHY bit rate [bit/s]

eta_bphz = Rb_gross_bps / B_occ_hz
# Spectral efficiency (gross PHY bits per second per Hz) [bit/s/Hz]

# Link budget
lambda_m = wavelength_m(f_hz)
# Wavelength [m]

L_fs_dB = fspl_db(R_m, f_hz)
# Free-space path loss [dB]

EIRP_dBW = eirp_dbw(P_tx_dBW, G_tx_dBi, L_tx_dB, L_point_tx_dB)
# Equivalent isotropically radiated power [dBW]

L_path_dB = total_path_loss_db(L_fs_dB, L_atm_dB, L_rain_dB, L_pol_dB, L_point_rx_dB, L_misc_dB)
# Total propagation/path loss [dB]

C_dBW = received_carrier_power_dbw(EIRP_dBW, L_path_dB, G_rx_dBi)
# Received carrier power at antenna terminals [dBW]

N0_dBW_per_Hz = noise_density_dbw_per_hz(T_sys_K)
# Noise density N0 [dBW/Hz]

CN0_dBHz = cn0_dbhz(C_dBW, N0_dBW_per_Hz)
# Carrier-to-noise-density ratio C/N0 [dB-Hz]

Rb_gross_dB = 10.0 * math.log10(Rb_gross_bps)
# 10log10(gross PHY bit rate) [dB]

EbN0_avail_dB = ebn0_db(CN0_dBHz, Rb_gross_bps, L_impl_dB)
# Available Eb/N0 [dB] using gross PHY bit rate

EbN0_avail_lin = db_to_lin(EbN0_avail_dB)
# Available Eb/N0 [linear]

BER = ber_qpsk_uncoded_awgn_from_ebn0_lin(EbN0_avail_lin)
# Predicted uncoded QPSK BER [-]

EbN0_req_lin, EbN0_req_dB = required_ebn0_for_target_ber(BER_target)
# Required Eb/N0 to meet BER_target [linear] and [dB]

Margin_dB = EbN0_avail_dB - EbN0_req_dB
# Eb/N0 link margin [dB]

Rb_max_bps = rb_max_for_target(CN0_dBHz, EbN0_req_dB, L_impl_dB)
# Maximum gross PHY bit rate that meets BER_target [bit/s]

# ============================================================
# OUTPUTS
# ============================================================

print("Link Budget + Throughput + BER (Uncoded coherent Gray QPSK, AWGN)")

print("\nKey intermediate values")
print(f"lambda [m]             : {lambda_m:.6e}")
print(f"L_fs [dB]              : {L_fs_dB:.3f}")
print(f"L_path [dB]            : {L_path_dB:.3f}")
print(f"EIRP [dBW]             : {EIRP_dBW:.3f}")
print(f"C [dBW]                : {C_dBW:.3f}")
print(f"N0 [dBW/Hz]            : {N0_dBW_per_Hz:.3f}")
print(f"C/N0 [dB-Hz]           : {CN0_dBHz:.3f}")
print(f"10log10(Rb_gross) [dB] : {Rb_gross_dB:.3f}")
print(f"Eb/N0 avail [dB]       : {EbN0_avail_dB:.3f}")
print(f"Eb/N0 req (target) [dB]: {EbN0_req_dB:.3f}")

print("\nOutputs")
print(f"BER        : {BER:.6e}")
print(f"Link margin [dB]       : {Margin_dB:.3f}")
print(f"Rb_max [bit/s]         : {Rb_max_bps:.6e}")

print("\nThroughput")
print(f"Rs [sym/s]             : {Rs_sps:.6e}")
print(f"Rb_gross [bit/s]       : {Rb_gross_bps:.6e}")
print(f"eta [bit/s/Hz]         : {eta_bphz:.6f}")