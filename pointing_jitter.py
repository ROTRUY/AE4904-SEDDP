import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
np.random.seed(1)

T = 5                 # total time [s]
fs = 100              # sampling frequency [Hz]
N = T * fs            # number of samples
w0 = 1.0              # normalized beam waist
sigma_jitter = 0.3    # pointing jitter std (normalized units)

# -----------------------------
# Time vector
# -----------------------------
time = np.linspace(0, T, N)

# -----------------------------
# 2D Gaussian pointing jitter
# -----------------------------
x_jitter = np.random.normal(0, sigma_jitter, N)
y_jitter = np.random.normal(0, sigma_jitter, N)

# Radial offset squared
r2 = x_jitter**2 + y_jitter**2

# -----------------------------
# Instantaneous normalized power
# Gaussian beam coupling model
# -----------------------------
P = np.exp(-2 * r2 / w0**2)

# -----------------------------
# Plot
# -----------------------------
detector_threshold = 0.2

plt.figure()
plt.plot(time, P, label="received power")
plt.axhline(np.mean(P), color="C1", linestyle="--", label="average received power")
plt.axhline(detector_threshold, color="C2", linestyle="--", label="detector threshold")
plt.xlabel("time (s)")
plt.ylabel("normalized power")
plt.title("Normalized Received Power vs Time")
plt.ylim(0, 1.05)
plt.legend()
plt.show()

# -----------------------------
# Estimated PDF (simulation)
# -----------------------------
bins = 70
pdf_sim, edges = np.histogram(P, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

# Theoretical PDF: f(I) = beta * I^(beta-1), beta = w0^2 / (4*sigma_jitter^2)
beta = w0**2 / (4 * sigma_jitter**2)
I_theory = np.linspace(0, 1, 500)
pdf_theory = beta * I_theory**(beta - 1)

plt.figure()
plt.plot(centers, pdf_sim, label="simulation")
plt.plot(I_theory, pdf_theory, color="red", label="theoretical")
plt.xlabel("normalized power")
plt.ylabel("probability density")
plt.title("PDF of Normalized Received Power")
plt.xlim(0, 1)
plt.legend()
plt.show()

# -----------------------------
# Statistics
# -----------------------------
outage_probability = np.sum(P < detector_threshold) / N
print("\n\nFinding outage probability...")
print("\nMean normalized power:", np.mean(P))
print("RMS pointing jitter (radial):", np.std(np.sqrt(r2)))
print("Outage probability:", outage_probability)