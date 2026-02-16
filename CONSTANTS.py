altitude = 550  # km
inclination = 98  # deg
dataVolume = 2.5  # Tb/day

RF Parameters:

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

# Waveform / throughput
B_occ_hz = 100e6
# Occupied (null-to-null) bandwidth used in design [Hz]

alpha = 0.2
# Roll-off factor

M = 4
# Constellation size (QPSK)

# BER / margin
BER_target = 1e-6
# Target bit error rate

L_impl_dB = 0.0
# Implementation loss applied at Eb/N0 level [dB]
