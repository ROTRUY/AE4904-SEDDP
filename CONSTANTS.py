altitude = 550  # km
inclination = 98  # deg
dataVolume = 2.5  # Tb/day

#RF Parameters:

# Geometry / frequency
f_hz = 1
# Carrier frequency [Hz]

R_m = 1
# Slant range between spacecraft and ground station [m]

# Transmit chain
P_tx_dBW = 1
# Transmit RF power at HPA output [dBW]

G_tx_dBi = 1
# Transmit antenna gain [dBi]

L_tx_dB = 1
# Transmit-side losses [dB]

L_point_tx_dB = 1
# Transmit pointing loss [dB]

# Propagation losses
L_atm_dB = 1
# Atmospheric/gaseous attenuation [dB]

L_rain_dB = 1
# Rain attenuation [dB]

L_pol_dB = 1
# Polarization mismatch loss [dB]

L_point_rx_dB = 1
# Receive pointing loss [dB]

L_misc_dB = 1
# Miscellaneous additional losses/margins in path [dB]

# Receiver
G_rx_dBi = 1
# Receive antenna gain [dBi]

T_sys_K = 1
# System noise temperature at antenna terminals [K]

# Waveform / throughput
B_occ_hz = 100e6
# Occupied (null-to-null) bandwidth [Hz]

alpha = 0.2
# Roll-off factor

M = 4
# Constellation size (QPSK)

# BER / margin
BER_target = 1e-6
# Target bit error rate

L_impl_dB = 0
# Implementation loss applied at Eb/N0 level [dB]
