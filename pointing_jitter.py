import numpy as np
import matplotlib.pyplot as plt
import optical_system as OS
import LinkBudgetOptical as LBO


class PointingJitterSimulation:
    """
    Monte Carlo simulation of pointing jitter fading for an optical downlink.
    """

    def __init__(self, optical_system: dict, T: float = 5.0, seed: int = 42):
        self._optical_system = optical_system
        self._link_budget = LBO.LinkBudget(optical_system)
        self.T = T
        self.seed = seed

        # --- Extract system parameters ---
        self.sigma_pj = self._link_budget.get_pointing_jitter()           # 1-axis RMS [rad]
        self.theta_div = self._link_budget.get_transmitter_divergence_angle()  # [rad]
        self.h = optical_system['link_benchmark_specs']['altitude']        # [m]
        self.elev = optical_system['link_benchmark_specs']['elevation_angle']  # [rad]
        self.f_max = optical_system['transmitter_specs']['fsm_bandwidth']  # [Hz]
        self.outage_threshold = optical_system['receiver_specs']['receiver_outage_power']  # normalised

        # --- Derived geometry ---
        self.R = self.h / np.sin(self.elev)     # slant range [m]
        self.w_r = self.theta_div * self.R      # beam radius at receiver [m]
        self.sigma_d = self.sigma_pj * self.R   # 1-axis displacement RMS [m]
        self.beta_param = self.w_r**2 / (4 * self.sigma_d**2)

        # --- Simulation time grid ---
        # fs = 2 * f_max satisfies Nyquist for fsm_bandwidth
        self.fs = 2 * self.f_max                # [Hz]
        self.N = int(self.T * self.fs)
        self.time = np.linspace(0, self.T, self.N)  # [s]

        rng = np.random.default_rng(self.seed)

        # Independent 1-axis Gaussian jitter [rad]
        self._theta_x = rng.normal(0, self.sigma_pj, self.N)
        self._theta_y = rng.normal(0, self.sigma_pj, self.N)

        # Transverse displacement at receiver [m]
        dx = self._theta_x * self.R
        dy = self._theta_y * self.R
        r2 = dx**2 + dy**2   # radial offset squared [m^2]

        # Gaussian beam intensity coupling
        self._P = np.exp(-2 * r2 / self.w_r**2)

    def get_jitter_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (theta_x, theta_y) time series [rad].
        """
        return self._theta_x, self._theta_y

    def get_normalised_power(self) -> np.ndarray:
        """
        Return normalised received power time series [-].
        """
        return self._P

    def get_mean_power(self) -> float:
        """
        Return mean normalised received power [-].
        """
        return float(np.mean(self._P))

    def get_outage_probability(self) -> float:
        """
        Return simulated outage probability: fraction of samples below outage_threshold [-].
        """
        return float(np.sum(self._P < self.outage_threshold) / self.N)

    def get_beta_param(self) -> float:
        """
        Return the beta distribution shape parameter [-].
        beta = w_r^2 / (4 * sigma_d^2)
        """
        return float(self.beta_param)

    def get_summary(self) -> dict:
        """
        Return a dict of key simulation statistics.
        """
        return {
            'sigma_pj_urad':       self.sigma_pj * 1e6,
            'beam_radius_m':       self.w_r,
            'sigma_displacement_m': self.sigma_d,
            'beta_param':          self.beta_param,
            'mean_power':          self.get_mean_power(),
            'outage_probability':  self.get_outage_probability(),
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_jitter_angles(self) -> None:
        """Plot angular jitter time series (theta_x, theta_y) vs time."""
        plt.figure()
        plt.plot(self.time * 1e3, self._theta_x * 1e6,
                 label=r'$\theta_x$', alpha=0.7, linewidth=0.8)
        plt.plot(self.time * 1e3, self._theta_y * 1e6,
                 label=r'$\theta_y$', alpha=0.7, linewidth=0.8)
        plt.axhline( self.sigma_pj * 1e6, color='gray', linestyle='--',
                     linewidth=1, label=r'$\pm\sigma_{pj}$')
        plt.axhline(-self.sigma_pj * 1e6, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Time [ms]')
        plt.ylabel(r'Angular jitter [$\mu$rad]')
        plt.title('Pointing Jitter — Angular Displacement vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_power_fading(self) -> None:
        """Plot normalised received power fading vs time."""
        plt.figure()
        plt.plot(self.time * 1e3, self._P, color='C0', linewidth=0.8,
                 label='Received power')
        plt.axhline(self.get_mean_power(), color='C1', linestyle='--', linewidth=1.5,
                    label=f'Mean power = {self.get_mean_power():.4f}')
        plt.axhline(self.outage_threshold, color='C2', linestyle='--', linewidth=1.5,
                    label=f'Outage threshold = {self.outage_threshold:.2f}')
        plt.xlabel('Time [ms]')
        plt.ylabel('Normalised received power [-]')
        plt.title('Normalised Received Power Fading due to Pointing Jitter')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_power_pdf(self, bins: int = 80) -> None:
        """
        Plot simulated PDF of normalised received power vs theoretical beta distribution.
        """
        pdf_sim, edges = np.histogram(self._P, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        I_theory = np.linspace(1e-6, 1, 500)
        pdf_theory = self.beta_param * I_theory ** (self.beta_param - 1)

        plt.figure()
        plt.plot(centers, pdf_sim, label='Simulation', linewidth=1.5)
        plt.plot(I_theory, pdf_theory, color='red', linestyle='--',
                 label=fr'Theory: $\beta$={self.beta_param:.1f}', linewidth=1.5)
        plt.xlabel('Normalised received power [-]')
        plt.ylabel('Probability density')
        plt.title('PDF of Received Power (Beta Distribution)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_all(self) -> None:
        """Generate all three plots and show them."""
        self.plot_jitter_angles()
        self.plot_power_fading()
        self.plot_power_pdf()
        plt.tight_layout()
        plt.show()

    def print_summary(self) -> None:
        """Print key simulation statistics to stdout."""
        s = self.get_summary()
        print(f"sigma_pj (1-axis):           {s['sigma_pj_urad']:.2f} urad")
        print(f"Beam radius at receiver:     {s['beam_radius_m']:.2f} m")
        print(f"sigma_displacement (1-axis): {s['sigma_displacement_m']:.4f} m")
        print(f"Beta parameter:              {s['beta_param']:.4f}")
        print(f"Mean normalised power:       {s['mean_power']:.6f}")
        print(f"Outage probability (sim):    {s['outage_probability']:.4e}")


if __name__ == '__main__':
    sim = PointingJitterSimulation(OS.optical_system1, T=5.0, seed=42)
    sim.print_summary()
    sim.plot_all()