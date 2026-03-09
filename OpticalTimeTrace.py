import numpy as np
import matplotlib.pyplot as plt
import math

# --- IMPORTS ---
import CONSTANTS as C
import optical_system as OS
from LinkBudgetOptical import LinkBudget 

# --- Setup Plotting ---
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

# --- 1. SETUP SYSTEM & GEOMETRY ---
specs = OS.optical_system1
alt_km = C.altitude
re_km = C.re
r_orb_km = re_km + alt_km
mu_km = C.mu                                   

w_sat = math.sqrt(mu_km / (r_orb_km**3))

# Pass duration calculation
el_min_rad = np.deg2rad(specs['receiver_specs'].get('min_elevation', 15)) 
lambda_max = math.acos((re_km / r_orb_km) * math.cos(el_min_rad)) - el_min_rad
pass_duration = 2 * (lambda_max / w_sat)

t = np.linspace(-pass_duration/2, pass_duration/2, 1000)
alpha_t = w_sat * t

# --- 2. CALCULATION LOOP ---
power_trace = []
atm_loss_trace = []
dist_trace_km = []

# LinkBudget class
lb = LinkBudget(specs)

for i in range(len(t)):
    # Calculate slant range in KM using Law of Cosines
    d_km = np.sqrt(re_km**2 + r_orb_km**2 - 2 * re_km * r_orb_km * np.cos(alpha_t[i]))
    
    # Calculate elevation for the current step
    current_el_rad = np.arccos((r_orb_km**2 - re_km**2 - d_km**2) / (2 * re_km * d_km))
    
    # --- CLASS UPDATE (Converting KM back to M for the internal functions) ---
    lb.R = d_km * 1000.0        # Range in meters for fspl/atm functions
    lb.elevation_angle = current_el_rad
    
    # Physics from your LinkBudgetOptical.py
    l_fs = lb.get_free_space_loss()
    l_atm = lb.get_atmospheric_loss()
    l_point = lb.get_static_pointing_error_loss() + lb.get_avg_pointing_jitter_loss()
    
    # Power Calculation (dBm)
    p_tx = specs['transmitter_specs']['transmitter_laser_power']
    g_tx = lb.get_transmitter_gain()
    g_rx = lb.get_receiver_gain()
    p_rx = p_tx + g_tx + g_rx - (l_fs + l_atm + l_point)
    
    # Append results
    dist_trace_km.append(d_km)
    power_trace.append(p_rx)
    atm_loss_trace.append(l_atm)

# --- 3. PLOTTING THE OPTICAL TRACE ---
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Optical Time Trace (KM Units) - Altitude: {alt_km} km", fontsize=14)

# Plot 1: Slant Range in KM
axs[0, 0].plot(t, dist_trace_km, 'b')
axs[0, 0].set_title("Slant Range ($d$)")
axs[0, 0].set_ylabel("Range [km]")

# Plot 2: Atmospheric Loss (dB)
axs[0, 1].plot(t, atm_loss_trace, 'g')
axs[0, 1].set_title("Atmospheric Loss")
axs[0, 1].set_ylabel("Loss [dB]")

# Plot 3: Elevation Angle
axs[1, 0].plot(t, np.degrees(np.arccos((r_orb_km**2 - re_km**2 - np.array(dist_trace_km)**2) / (2 * re_km * np.array(dist_trace_km)))), 'r')
axs[1, 0].set_title("Elevation Angle")
axs[1, 0].set_ylabel("Elevation [deg]")

# Plot 4: Received Power (dBm)
axs[1, 1].plot(t, power_trace, 'm', linewidth=2)
axs[1, 1].set_title("Received Power ($P_{rx}$)")
axs[1, 1].set_ylabel("Power [dBm]")

for ax in axs.flat:
    ax.grid(True)
    ax.set_xlabel("Time from TCA [s]")

plt.tight_layout()
plt.show()