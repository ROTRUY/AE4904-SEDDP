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


def plot_ber_contour(
    aperture_range: np.ndarray,
    power_range: np.ndarray,
    ber_grid: np.ndarray,
    ber_targets: list = [1e-6, 1e-3],
) -> None:
    D_mesh, P_mesh = np.meshgrid(aperture_range * 100, power_range, indexing='ij')
    log_ber = np.log10(np.clip(ber_grid, 1e-20, 1.0))
    colors = ['royalblue', 'crimson']

    plt.figure(figsize=(8, 6))

    paths = {}
    for ber_target, color in zip(ber_targets, colors):
        log_target = np.log10(ber_target)
        plt.contour(D_mesh, P_mesh, log_ber, levels=[log_target], colors=color, linestyles='solid')

    # Proxy lines for legend
    proxy_lines = [
        plt.Line2D([0], [0], color=c, linewidth=1.8, linestyle='solid',
                   label=f'BER = {t:.0e}')
        for t, c in zip(ber_targets, colors)
    ]
    # Adapt axis to contour vertices
    all_v = np.vstack(list(paths.values())) if paths else None
    if all_v is not None:
        xm = (all_v[:, 0].max() - all_v[:, 0].min()) * 0.05
        ym = (all_v[:, 1].max() - all_v[:, 1].min()) * 0.05
        plt.xlim(all_v[:, 0].min() - xm, all_v[:, 0].max() + xm)
        plt.ylim(all_v[:, 1].min() - ym, all_v[:, 1].max() + ym)

    plt.legend(handles=proxy_lines, fontsize=9)
    plt.xlabel('Transmitter aperture $D_T$ [cm]')
    plt.ylabel('Laser power [dBm]')
    plt.title('Required aperture vs laser power — feasible region')
    plt.grid(True, alpha=0.3)
    # plt.axis([5,20,20,35])
    plt.tight_layout()
    plt.show()


def plot_ber_3d(
    aperture_range: np.ndarray,
    power_range: np.ndarray,
    ber_grid: np.ndarray
) -> None:
    """
    3D surface plot of log10(BER) over transmitter aperture and laser power.
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
    plt.clf()
    N = 10
    aperture_range = np.linspace(0.05, 0.40, N)  # 5 cm to 40 cm
    power_range = np.linspace(5, 40, N)  # 5 dBm to 40 dBm (10 W)
    ber_targets = np.array([1e-6, 1e-3])
    ber_target = 1e-6

    print("Running BER grid sweep...")
    ber_grid = compute_ber_grid(
        aperture_range, power_range,
        base_system=OS.optical_system1,
        T_sim=1.0,
        seed=42,
    )

    # plot_ber_3d(aperture_range, power_range, ber_grid)
    plot_ber_contour(aperture_range, power_range, ber_grid, ber_targets)
