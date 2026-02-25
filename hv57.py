import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# HV57 Turbulence Model
# -----------------------------

# Parameters for HV57
A = 1.7e-14        # m^(-2/3)
v = 21             # m/s
h0 = 0             # ground level reference altitude (m)

# Altitude range (0 to 20 km)
h = np.linspace(0, 20000, 1000)   # meters

# HV57 model equation
Cn2 = (
    A * np.exp(-h/700) * np.exp(-(h - h0)/100)
    + 2.7e-16 * np.exp(-h/1500)
    + 0.00594 * (v/27)**2 * (h * 1e-5)**10 * np.exp(-h/1000)
)

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(6, 8))
plt.semilogx(Cn2, h/1000, 'b', linewidth=2)

plt.xlabel(r"Turbulence strength, $C_n^2$")
plt.ylabel("Altitude [km]")
plt.title("HV57 Atmospheric Turbulence Profile")

plt.xlim(1e-18, 1e-14)
plt.ylim(0, 20)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()