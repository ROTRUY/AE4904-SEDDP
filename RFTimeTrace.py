import numpy as np
import matplotlib.pyplot as plt
import math

import RFBERLinkBudgetThroughput as LB
import CONSTANTS as C

# --- Setup Plotting for Spyder (Internal MathText) ---
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})


# --- System Parameters ---
re_km = C.re
mu = C.mu
alt_km = C.altitude
k_db = C.k_db
c = C.c
f_hz = C.f_hz
P_tx_dBW = C.P_tx_dBW
G_tx_dBi = C.G_tx_dBi
G_rx_dBi = C.G_rx_dBi
T_sys_K = C.T_sys_K
Rb_bps = LB.Rb_gross_bps 


# --- Geometry Trace ---
r_orb = re_km + alt_km
w_sat = math.sqrt(mu / (r_orb**3)) # angular velocity
el_min = np.deg2rad(5) # minimal elevation
lambda_max = math.acos((re_km / r_orb) * math.cos(el_min)) - el_min 
# max earth central angle
pass_duration = 2 * (lambda_max / w_sat) # time of passage

t = np.linspace(-pass_duration/2, pass_duration/2, 1000)
alpha_t = w_sat * t # angle between GS and satellite
d_km = np.sqrt(re_km**2 + r_orb**2 - 2 * re_km * r_orb * np.cos(alpha_t))
# slant range
el_deg = np.degrees(np.arccos((r_orb**2 - re_km**2 - d_km**2) / (2 * re_km * d_km)))
# elevation angle
R_m = d_km * 1000

# --- Calculation Loop ---
ebno_trace, fspl_trace, doppler_trace = [], [], []
EIRP = LB.EIRP_dBW
N0 = LB.N0_dBW_per_Hz

for i in range(len(t)):
    L_fs = LB.fspl_db(R_m[i], f_hz)
    CN0 = (EIRP - (L_fs + 2.0) + G_rx_dBi) - N0 # +2dB misc path loss
    v_r = (re_km * r_orb * w_sat * np.sin(alpha_t[i])) / d_km[i] # radial velocity

    fspl_trace.append(L_fs)
    ebno_trace.append(LB.ebn0_db(CN0, Rb_bps, 1.5))
    doppler_trace.append(-(f_hz / c) * (v_r * 1000.0))

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"RF Downlink Time Trace: {f_hz/1e9:.1f} GHz", fontsize=14)

axs[0, 0].plot(t, el_deg, 'b'); axs[0, 0].set_title("Elevation Angle"); axs[0, 0].set_ylabel("Deg ($^\circ$)")
axs[0, 1].plot(t, np.array(doppler_trace)/1e3, 'r'); axs[0, 1].set_title("Doppler Shift"); axs[0, 1].set_ylabel("kHz")
axs[1, 0].plot(t, fspl_trace, 'g'); axs[1, 0].set_title("Free Space Path Loss"); axs[1, 0].set_ylabel("dB")
axs[1, 1].plot(t, ebno_trace, 'm'); axs[1, 1].set_title("Available $E_b/N_0$"); axs[1, 1].set_ylabel("dB")

for ax in axs.flat: ax.grid(True); ax.set_xlabel("Time from TCA (s)")
plt.tight_layout(); plt.show()