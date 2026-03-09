from math import pi
import numpy as np
import matplotlib.pyplot as plt
import optical_system as OS
import LinkBudgetOptical as LBO

# Assumptions:
#   - Jitter is zero-mean, isotropic 2D Gaussian (independent x/y axes)
#   - sigma_pj = fsm_jitter (1-axis RMS, post-FSM residual) [rad]
#   - Angular jitter translates to transverse displacement at receiver: d = theta * R
#   - Beam radius at receiver: w_r = theta_div * R (far-field approximation)
#   - Simulation duration T and sampling rate fs chosen to resolve jitter dynamics
#     up to fsm_bandwidth (1000 Hz) — fs must be >= 2 * fsm_bandwidth (Nyquist)

np.random.seed(42)

# --- System parameters from optical_system.py ---
optical_system = OS.optical_system1
link_budget = LBO.LinkBudget(optical_system)

sigma_pj = link_budget.get_pointing_jitter()
Lambda = optical_system['transmitter_specs']['system_frequency'] # m
theta_div = optical_system['transmitter_specs']['transmitter_divergence_angle']  # rad
h = optical_system['link_benchmark_specs']['altitude']  # m
elev = optical_system['link_benchmark_specs']['elevation_angle']  # rad
f_max = optical_system['transmitter_specs']['fsm_bandwidth']  # Hz
outage = optical_system['receiver_specs']['receiver_outage_power']  # normalised power threshold

R = h / np.sin(elev)   # slant range [m]
w_r = theta_div * R    # beam radius at receiver [m], far-field approximation

# --- Simulation time parameters ---
# fs must satisfy Nyquist: fs >= 2 * f_max
fs = 2 * f_max    # Hz — minimum Nyquist rate for fsm_bandwidth
T = 5.0  # s — total simulation time
N = int(T * fs)
time = np.linspace(0, T, N)

# --- 2D Gaussian pointing jitter [rad] ---
# sigma_pj is the 1-axis angular RMS; displacement at receiver = angle * R
theta_x = np.random.normal(0, sigma_pj, N)  # rad, x-axis
theta_y = np.random.normal(0, sigma_pj, N)  # rad, y-axis

# Transverse displacement at receiver [m]
dx = theta_x * R
dy = theta_y * R
r2 = dx**2 + dy**2   # radial offset squared [m^2]

# Radial angular offset [rad]
r_angular = np.sqrt(theta_x**2 + theta_y**2)

# --- Normalised received power (Gaussian beam coupling) ---
P = np.exp(-2 * r2 / w_r**2)

# --- Jitter angle time series ---
plt.figure()
plt.plot(time * 1e3, theta_x * 1e6, label=r'$\theta_x$', alpha=0.7, linewidth=0.8)
plt.plot(time * 1e3, theta_y * 1e6, label=r'$\theta_y$', alpha=0.7, linewidth=0.8)
plt.axhline(sigma_pj * 1e6, color='gray', linestyle='--', linewidth=1, label=r'$\pm\sigma_{pj}$')
plt.axhline(-sigma_pj * 1e6, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Time [ms]')
plt.ylabel(r'Angular jitter [$\mu$rad]')
plt.title('Pointing Jitter — Angular Displacement vs Time')
plt.legend()
plt.grid(True, alpha=0.3)

# --- Normalised received power vs time ---
plt.figure()
plt.plot(time * 1e3, P, color='C0', linewidth=0.8, label='Received power')
plt.axhline(np.mean(P), color='C1', linestyle='--', linewidth=1.5,
                label=f'Mean power = {np.mean(P):.4f}')
plt.axhline(outage, color='C2', linestyle='--', linewidth=1.5,
                label=f'Outage threshold = {outage:.2f}')
plt.xlabel('Time [ms]')
plt.ylabel('Normalised received power [-]')
plt.title('Normalised Received Power Fading due to Pointing Jitter')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)

# --- PDF of received power ---
bins = 80
pdf_sim, edges = np.histogram(P, bins=bins, density=True)
centers = 0.5 * (edges[:-1] + edges[1:])

# --- Theoretical PDF: beta distribution ---
sigma_d = sigma_pj * R          # 1-axis displacement RMS [m]
w_0 = Lambda / (pi * theta_div)
beta_param = w_0**2 / (4 * sigma_d**2)
I_theory = np.linspace(1e-6, 1, 500)   # avoid log(0)
pdf_theory = beta_param * I_theory ** (beta_param - 1)

plt.figure()
plt.plot(centers, pdf_sim, label='Simulation', linewidth=1.5)
plt.plot(I_theory, pdf_theory, color='red', linestyle='--',
             label=fr'Theory: $\beta$={beta_param:.1f}', linewidth=1.5)
plt.xlabel('Normalised received power [-]')
plt.ylabel('Probability density')
plt.title('PDF of Received Power (Beta Distribution)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Statistics ---
outage_prob_sim = np.sum(P < outage) / N
print(f"sigma_pj (1-axis):          {sigma_pj*1e6:.2f} urad")
print(f"Beam radius at receiver:    {w_r:.2f} m")
print(f"sigma_displacement (1-axis):{sigma_d:.4f} m")
print(f"beta parameter:             {beta_param:.4f}")
print(f"Mean normalised power:      {np.mean(P):.6f}")
print(f"Outage probability (sim):   {outage_prob_sim:.4e}")