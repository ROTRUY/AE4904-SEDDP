# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

import optical_system as OS
import LinkBudgetOptical as LBO
from NoiseOptical import OpticalNoiseV2
from pointing_jitter import PointingJitterSimulation

# Constants
R_b = 2e9  # bit rate [bit/s]
B_e = R_b / 2  # electrical receiver bandwidth (= 1 GHz)
seed = 42


class BERSimulation:

    def __init__(self, optical_system: dict, T_sim: float = 10.0):
        self.os = optical_system
        self.lb = LBO.LinkBudget(optical_system)
        self.T_sim = T_sim

        # --- 1. Mean received power from link budget ---
        tx_power_dBm = optical_system['transmitter_specs']['transmitter_laser_power']
        total_dBm = self.lb.get_total_link_budget(tx_power_dBm)
        self.P_R = 10 ** ((total_dBm - 30) / 10)  # [W]

        # --- 2. Noise model at mean received power ---
        self.wavelength = optical_system['transmitter_specs']['wavelength']
        self.noise = OpticalNoiseV2(self.P_R, self.wavelength, B_e)
        self.SNR = self.noise.SNR
        self.sigma2 = self.noise.sigmaTotal  # noise variance [A²]

        # --- 3. Pointing jitter fading trace ---
        self.jitter_sim = PointingJitterSimulation(self.os, self.T_sim, seed)
        self.jitter_sim.run()

        self.u, self.n_raw, self.t = self._generate_traces()
        self.n = self._apply_lpf(self.n_raw)

        self.BER_mean = self.compute_mean_BER()

    def _generate_scintillation(self, N: int, dt: float) -> np.ndarray:
        """
        Generate a normalised log-normal scintillation fading trace.

        u_scint[k] = exp( x[k] )
        x[k] = rho * x[k-1] + sqrt(1 - rho^2) * sigma_ln * w[k]
        rho   = exp(-dt / tau_c)
        sigma_ln^2 = log(1 + sigma_I^2)
        """
        rng = np.random.default_rng(seed + 2)
        tau_c = self.os['transmitter_specs']['tau_c']
        sigma_I2 = self.lb.get_scintillation_index()
        sigma_ln = np.sqrt(np.log(1 + sigma_I2))   # log-normal parameter
        rho = np.exp(- dt / tau_c)

        x = np.zeros(N)
        w = rng.normal(0.0, 1.0, N)
        for k in range(1, N):
            x[k] = rho * x[k-1] + np.sqrt(1 - rho**2) * sigma_ln * w[k]

        u_scint = np.exp(x)
        u_scint /= np.mean(u_scint)   # normalise to unit mean
        return u_scint

    def _generate_traces(self):
        """
        u : ndarray  — normalised fading trace from pointing jitter [-]
        n : ndarray  — white Gaussian noise trace, variance = sigma2 [A]
        t : ndarray  — time vector [s]
        """
        rng = np.random.default_rng(seed + 1)

        self.u_jitter = self.jitter_sim.get_normalised_power()
        t = self.jitter_sim.time
        dt = t[1] - t[0]
        N = len(self.u_jitter)

        self.u_scint = self._generate_scintillation(N, dt)
        u = self.u_scint * self.u_jitter
        u /= np.mean(u)  # normalise combined trace to unit mean

        n = rng.normal(0.0, np.sqrt(self.sigma2), size=N)

        return u, n, t

    def _apply_lpf(self, n: np.ndarray) -> np.ndarray:
        """
        Low-pass filter the noise trace at cutoff B_e = R_b / 2.

        Since n(t) is sampled at fs = 2*fsm_bandwidth = 2 kHz, which is far
        below B_e = 1 GHz, the filter has no effect at this sampling rate.
        The bandwidth B_e is already embedded in sigma2 via OpticalNoiseV2.

        n : ndarray  — white Gaussian noise trace [A]
        n_filtered : ndarray  — filtered noise trace [A]
        """
        fs = 2 * self.os['transmitter_specs']['fsm_bandwidth']  # 2 kHz
        kernel_width = max(1, int(fs / B_e))                    # = 1

        if kernel_width == 1:
            return n.copy()

        kernel = np.ones(kernel_width) / kernel_width
        return np.convolve(n, kernel, mode='same')

    def _select_windows(self, N_windows: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide the fading and noise traces into N_windows non-overlapping
        windows of equal length.
        """
        N_total = len(self.u)
        N_samples = N_total // N_windows  # samples per window

        N_used = N_windows * N_samples  # total samples actually used

        self.N_windows = N_windows

        u_cut = self.u[:N_used]
        n_cut = self.n[:N_used]

        u_windows = u_cut.reshape(N_windows, N_samples)
        n_windows = n_cut.reshape(N_windows, N_samples)

        return u_windows, n_windows

    def _compute_window_BER(self, u_windows: np.ndarray) -> np.ndarray:
        """
        Compute BER for each window using the mean fading level.
        """
        u_k = np.mean(u_windows, axis=1)  # shape (N_windows,)

        BER_windows = 0.5 * erfc(np.sqrt(self.SNR) * u_k / (2 * np.sqrt(2)))

        return BER_windows

    def compute_mean_BER(self, N_windows: int = 100) -> float:
        """
        Compute the averaged BER over all windows.

        <BER> = (1/N_windows) * sum_k BER(u_k)
            ≈ integral_0^inf p(u) * 0.5 * erfc(<SNR>*u / 2*sqrt(2)) du
        """
        u_windows, n_windows = self._select_windows(N_windows)
        BER_windows = self._compute_window_BER(u_windows)
        self.BER_mean = float(np.mean(BER_windows))
        self.BER_windows = BER_windows   # store for plots
        self.u_k = np.mean(u_windows, axis=1)  # store for plots

        return self.BER_mean

    def compute_BER_sweep(self, P_R_min_dBm: float = -40.0, P_R_max_dBm: float = -10.0, N_points: int = 50):
        """
        Sweep received power P_R and compute <BER> at each point.
        The fading trace u(t) is fixed; only the noise model is updated.
        """
        P_R_dBm_arr = np.linspace(P_R_min_dBm, P_R_max_dBm, N_points)
        SNR_dB = np.zeros(N_points)
        BER_arr = np.zeros(N_points)

        u_windows, _ = self._select_windows(self.N_windows)

        for i, P_R_dBm in enumerate(P_R_dBm_arr):
            P_R_i = 10 ** ((P_R_dBm - 30) / 10)   # [W]

            noise_i = OpticalNoiseV2(P_R_i, self.wavelength, B_e)
            SNR_i = np.sqrt(noise_i.SNR)  # amplitude ratio
            BER_arr[i] = np.mean(0.5 * erfc(SNR_i * np.mean(u_windows, axis=1) / (2 * np.sqrt(2))))
            SNR_dB[i] = 10 * np.log10(noise_i.SNR)  # power ratio in dB

        self.sweep_SNR_dB = SNR_dB
        self.sweep_BER = BER_arr

        return SNR_dB, BER_arr

    def print_summary(self) -> None:
        """Print key simulation parameters and the averaged BER."""
        print("=" * 45)
        print("        BER Simulation Summary")
        print("=" * 45)
        print(f"  Mean received power P_R : {self.P_R:.3e} W  "
              f"({10*np.log10(self.P_R*1e3):.2f} dBm)")
        print(f"  Mean SNR                : {10*np.log10(self.SNR):.2f} dB")
        print(f"  Noise variance sigma2   : {self.sigma2:.3e} A^2")
        print(f"  Beta parameter          : {self.jitter_sim.get_beta_param():.4f}")
        print(f"  Mean fading <u>         : {float(np.mean(self.u)):.4f}")
        print(f"  N windows               : {self.N_windows}")
        print(f"  Samples per window      : {len(self.u) // self.N_windows}")
        print(f"  Static BER (no fading)  : "
              f"{0.5*erfc(np.sqrt(self.SNR/2)):.2e}")
        print(f"  <BER> (fading average)  : {self.BER_mean:.2e}")
        print("=" * 45)

    def plot_scintillation(self) -> None:
        plt.figure()
        plt.plot(self.t * 1e3, self.u_scint, color='steelblue', linewidth=0.8)
        plt.axhline(np.mean(self.u_scint), color='darkorange', linestyle='--',
                    linewidth=1.5, label=f'Mean = {np.mean(self.u_scint):.3f}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Normalised power [–]')
        plt.title('Scintillation Fading $u_{scint}(t)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


    def plot_jitter(self) -> None:
        plt.figure()
        plt.plot(self.t * 1e3, self.u_jitter, color='steelblue', linewidth=0.8)
        plt.axhline(np.mean(self.u_jitter), color='darkorange', linestyle='--',
                    linewidth=1.5, label=f'Mean = {np.mean(self.u_jitter):.4f}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Normalised power [–]')
        plt.title('Pointing Jitter Fading $u_{jitter}(t)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


    def plot_combined_fading(self) -> None:
        plt.figure()
        plt.plot(self.t * 1e3, self.u, color='steelblue', linewidth=0.8)
        plt.axhline(np.mean(self.u), color='darkorange', linestyle='--',
                    linewidth=1.5, label=f'Mean = {np.mean(self.u):.4f}')
        plt.axhline(self.os['receiver_specs']['receiver_outage_power'],
                    color='crimson', linestyle='--', linewidth=1.5,
                    label=f'Outage threshold = {self.os["receiver_specs"]["receiver_outage_power"]}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Normalised power [–]')
        plt.title('Combined Fading $u(t) = u_{scint}(t) \cdot u_{jitter}(t)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_noise(self) -> None:
        plt.figure()
        plt.plot(self.t * 1e3, self.n, color='steelblue', linewidth=0.8)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axhline( 3*np.sqrt(self.sigma2), color='crimson', linestyle='--',
                    linewidth=1.5, label=r'$\pm 3\sigma$')
        plt.axhline(-3*np.sqrt(self.sigma2), color='crimson', linestyle='--',
                    linewidth=1.5)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current [A]')
        plt.title('Noise Trace $n(t)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_BER_trace(self) -> None:
        u_windows, _ = self._select_windows(self.N_windows)
        BER_windows = self._compute_window_BER(u_windows)
        t_windows = np.linspace(self.t[0] * 1e3, self.t[-1] * 1e3, self.N_windows)

        plt.figure()
        plt.semilogy(t_windows, BER_windows, color='steelblue',
                    linewidth=1.5, marker='o', markersize=3)
        plt.axhline(1e-6, color='crimson', linestyle='--',
                    linewidth=1.5, label='BER = $10^{-6}$')
        plt.axhline(self.BER_mean, color='darkorange', linestyle='--',
                    linewidth=1.5, label=f'$\\langle$BER$\\rangle$ = {self.BER_mean:.2e}')
        plt.xlabel('Time [ms]')
        plt.ylabel('BER')
        plt.title('BER Time Trace BER$(u(t))$')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()

    def plot_BER_sweep(self) -> None:
        SNR_dB, BER_arr = self.compute_BER_sweep()
        mask = BER_arr > 0

        plt.figure()
        plt.semilogy(SNR_dB[mask], BER_arr[mask], color='steelblue', linewidth=2)
        plt.axhline(1e-6, color='crimson', linestyle='--',
                    linewidth=1.5, label='BER = $10^{-6}$')
        plt.axvline(10 * np.log10(self.SNR), color='darkorange', linestyle='--',
                    linewidth=1.5,
                    label=f'Operating point ({10*np.log10(self.SNR):.1f} dB)')
        plt.xlabel('Mean SNR [dB]')
        plt.ylabel('BER')
        plt.title('BER vs SNR')
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()

    def plot_fading_pdf(self) -> None:
        beta = self.jitter_sim.get_beta_param()
        u_theory = np.linspace(1e-4, max(self.u), 500)
        pdf_sim, edges = np.histogram(self.u, bins=80, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.figure()
        plt.bar(centers, pdf_sim, width=np.diff(edges),
                alpha=0.5, color='steelblue', label='Simulation')
        plt.plot(u_theory, beta * u_theory ** (beta - 1), color='crimson',
                linestyle='--', linewidth=2,
                label=fr'Beta PDF $\beta$={beta:.1f}')
        plt.xlabel('Normalised power $u$ [–]')
        plt.ylabel('Probability density')
        plt.title('PDF of Fading $p(u)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_all(self) -> None:
        self.plot_scintillation()
        self.plot_jitter()
        self.plot_combined_fading()
        self.plot_noise()
        # self.plot_BER_trace()
        self.plot_BER_sweep()
        self.plot_fading_pdf()
        plt.show()


if __name__ == '__main__':
    sim = BERSimulation(OS.optical_system1, T_sim=10.0)
    sim.print_summary()
    sim.plot_all()