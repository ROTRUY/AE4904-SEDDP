### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

### SIMULATION
# Parameters
fs = 1000          # sampling frequency [Hz]
T = 10             # duration [s]
theta_div = 50e-6  # beam divergence [µrad]

# Set up time and frequency vectors
N = int(T * fs)
dt = 1/fs
t = np.arange(N) * dt
f = np.fft.rfftfreq(N, dt)

# Jitter PSD
S = 160 / (1 + f**2)

# Generate complex white noise
random_complex = (
    np.random.randn(len(f)) +
    1j * np.random.randn(len(f))
)

# Shape according to PSD
Theta_f = random_complex * np.sqrt(S * fs / 2)

# Inverse FFT to get time-domain jitter
theta = np.fft.irfft(Theta_f, n=N) * 1e-6  # rad


# Get power loss from pointing error
I_point = np.exp(-2 * (theta**2) / theta_div**2)

# Turbulence parameters
scint_index = 0.1         # scintillation index (from slide)
fc_turb = 120          # turbulence bandwidth [Hz]

# Convert scintillation index → lognormal parameters
sigma_ln = np.sqrt(np.log(scint_index**2 + 1))
mu_ln = - sigma_ln**2 / 2

# Generate Gaussian noise
nz = np.random.randn(N)

# Band-limit turbulence
b, a = butter(2, fc_turb/(fs/2))
Z = filtfilt(b, a, nz)

# Normalize variance
Z = Z / np.std(Z)

# Log-normal intensity
I_turb = np.exp(mu_ln + sigma_ln * Z)

# Total
I_total = I_point * I_turb

### PLOT
plt.figure(figsize=(10,8))

plt.subplot(4,1,1)
plt.plot(t, theta*1e6)
plt.ylabel("Jitter [µrad]")

plt.subplot(4,1,2)
plt.plot(t, I_point)
plt.ylabel("Pointing")

plt.subplot(4,1,3)
plt.plot(t, I_turb)
plt.ylabel("Turbulence")

plt.subplot(4,1,4)
plt.plot(t, I_total)
plt.ylabel("Total")
plt.xlabel("Time [s]")

plt.tight_layout()
plt.show()

### STATS
print("RMS Jitter [µrad]:", np.std(theta)*1e6)
print("Mean Turbulence:", np.mean(I_turb))
print("Mean Total Power:", np.mean(I_total))