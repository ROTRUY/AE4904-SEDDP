"""
3D sweep of mean BER as a function of transmitter aperture and laser power.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

import optical_system as OS
from BER import BERSimulation


def compute_ber_grid(
    aperture_range: np.ndarray,
    power_range: np.ndarray,
    base_system: dict,
    T_sim: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute mean BER on a 2D grid of (transmitter_aperture, laser_power).
    """
    ber_grid = np.zeros((len(aperture_range), len(power_range)))

    for i, D_T in enumerate(aperture_range):
        for j, P_tx in enumerate(power_range):
            sys = copy.deepcopy(base_system)
            sys['transmitter_specs']['transmitter_aperture'] = D_T
            sys['transmitter_specs']['transmitter_laser_power'] = P_tx

            try:
                sim = BERSimulation(sys, T_sim=T_sim)
                ber_grid[i, j] = sim.BER_mean
            except Exception as e:
                print(f"  Failed at D_T={D_T:.3f} m, P_tx={P_tx:.1f} dBm: {e}")
                ber_grid[i, j] = np.nan
    return ber_grid


def find_optimum(
    aperture_range: np.ndarray,
    power_range: np.ndarray,
    ber_grid: np.ndarray,
    ber_target: float = 1e-6,
) -> float:
    """
    Find optimum operating point from the precomputed BER grid.

    Hardware-efficient minimising the combined normalised cost:
            cost = (D_T - D_T_min) / (D_T_max - D_T_min)
                 + (P_tx - P_tx_min) / (P_tx_max - P_tx_min)
        i.e. the point with smallest aperture and lowest power that meets
        the target. Equal weighting assumed.
    """
    valid = ~np.isnan(ber_grid)

    # --- Criterion: smallest hardware footprint meeting BER target ---
    feasible = valid & (ber_grid <= ber_target)
    if not np.any(feasible):
        result = None
    else:
        D_norm = (aperture_range - aperture_range.min()) / (aperture_range.max() - aperture_range.min())
        P_norm = (power_range - power_range.min()) / (power_range.max() - power_range.min())
        D_mesh, P_mesh = np.meshgrid(D_norm, P_norm, indexing='ij')
        cost = 0.8*D_mesh + 0.2*P_mesh      # equal-weight normalised sum
        cost[~feasible] = np.inf
        idx_flat2 = np.argmin(cost)
        i2, j2 = np.unravel_index(idx_flat2, ber_grid.shape)
        result = {
            'D_T':  aperture_range[i2],
            'P_tx': power_range[j2],
            'BER':  ber_grid[i2, j2],
            'i': i2, 'j': j2,
        }

    return result


def print_optimum(optimum: float, ber_target: float = 1e-6) -> None:
    r = optimum
    print("\nCriterion 2 — Hardware-efficient (BER <= {:.0e})".format(ber_target))
    if r is None:
        print("  No grid point meets the BER target.")
    else:
        print(f"  D_T     : {r['D_T']*100:.2f} cm")
        print(f"  P_tx    : {r['P_tx']:.1f} dBm")
        print(f"  BER     : {r['BER']:.3e}")
    print("=" * 50)


def plot_ber_3d(
    aperture_range: np.ndarray,
    power_range: np.ndarray,
    ber_grid: np.ndarray,
    optimum: dict = None,
) -> None:
    """
    3D surface plot of log10(BER) over transmitter aperture and laser power.
    Optionally marks criterion 1 (star) and criterion 2 (diamond) optimum points.
    """
    D_mesh, P_mesh = np.meshgrid(aperture_range * 100, power_range, indexing='ij')
    log_ber = np.log10(np.clip(ber_grid, 1e-20, 1.0))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        D_mesh, P_mesh, log_ber,
        cmap='plasma', edgecolor='none', alpha=0.85
    )

    # BER = 1e-6 target plane
    ax.plot_surface(
        D_mesh, P_mesh,
        np.full_like(log_ber, -6),
        alpha=0.2, color='cyan'
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, label='$\\log_{10}$(BER)')
    ax.set_xlabel('Transmitter aperture $D_T$ [cm]')
    ax.set_ylabel('Laser power [dBm]')
    ax.set_zlabel('$\\log_{10}$(BER)')
    ax.set_title('Mean BER vs $D_T$ and laser power\n(cyan plane: BER = $10^{-6}$)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    aperture_range = np.linspace(0.05, 0.20, 50)  # 5 cm to 20 cm
    power_range = np.linspace(5, 40, 50)  # 20 dBm (100 mW) to 40 dBm (10 W)

    print("Running BER grid sweep...")
    ber_grid = compute_ber_grid(
        aperture_range, power_range,
        base_system=OS.optical_system1,
        T_sim=1.0,
        seed=42,
    )

    optimum = find_optimum(aperture_range, power_range, ber_grid, ber_target=1e-6)
    print_optimum(optimum, ber_target=1e-6)

    plot_ber_3d(aperture_range, power_range, ber_grid, optimum=optimum)