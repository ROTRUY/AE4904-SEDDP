import numpy as np
import matplotlib.pyplot as plt

import optical_system as OS


def main(optical_system: dict = OS.optical_system1) -> None:
    elevation_angle = optical_system["link_benchmark_specs"]["elevation_angle"]  # [rad]
    altitude = optical_system["link_benchmark_specs"]["altitude"]  # [m]
    wavelength = optical_system["transmitter_specs"]["wavelength"]  # [m]

    transmitter_pointing_error = optical_system["transmitter_specs"]["transmitter_pointing_error"]  # [rad]
    sigma_pj = optical_system["transmitter_specs"]["fsm_accuracy"]  # [rad]
    outage_probability = optical_system["receiver_specs"].get("outage_probability", 1e-3)

    link_range = altitude / np.sin(elevation_angle)  # [m]

    theta_div_urad = np.linspace(0.01, 40.0, 1000)  # sweep [µrad] (avoid 0 to prevent divide-by-zero)
    theta_div = theta_div_urad * 1e-6  # [rad]

    free_space_loss_linear = (4 * np.pi * link_range / wavelength) ** 2
    free_space_loss_db = -10 * np.log10(free_space_loss_linear)

    gain_tx_linear = 8 / (theta_div**2)
    gain_tx_db = 10 * np.log10(gain_tx_linear)
    geometric_loss_db = free_space_loss_db + gain_tx_db

    static_pointing_error_loss_db = (-20.0 / np.log(10.0)) * (
        transmitter_pointing_error**2 / theta_div**2
    )

    exponent = 4.0 * sigma_pj**2 / theta_div**2
    pointing_jitter_scintillation_loss_db = 10.0 * exponent * np.log10(outage_probability)

    pointing_loss_db = static_pointing_error_loss_db + pointing_jitter_scintillation_loss_db
    combined_loss_db = geometric_loss_db + pointing_loss_db

    theta_print_urad = np.arange(1.0, 11.0, 1.0)
    theta_print = theta_print_urad * 1e-6

    geometric_loss_db_print = free_space_loss_db + 10.0 * np.log10(8.0 / (theta_print**2))
    static_pointing_error_loss_db_print = (-20.0 / np.log(10.0)) * (
        transmitter_pointing_error**2 / theta_print**2
    )
    exponent_print = 4.0 * sigma_pj**2 / theta_print**2
    pointing_jitter_scintillation_loss_db_print = 10.0 * exponent_print * np.log10(outage_probability)
    pointing_loss_db_print = (
        static_pointing_error_loss_db_print + pointing_jitter_scintillation_loss_db_print
    )
    combined_loss_db_print = geometric_loss_db_print + pointing_loss_db_print

    print("\nValues at integer divergence angles (1–10 µrad):")
    print(f"{'θ_div [µrad]':>11}  {'L_geom [dB]':>12}  {'L_point [dB]':>13}  {'L_comb [dB]':>12}")
    for i in range(len(theta_print_urad)):
        print(
            f"{theta_print_urad[i]:11.0f}  {geometric_loss_db_print[i]:12.2f}  "
            f"{pointing_loss_db_print[i]:13.2f}  {combined_loss_db_print[i]:12.2f}"
        )

    idx_max = int(np.argmax(combined_loss_db))
    theta_div_max_urad = float(theta_div_urad[idx_max])
    combined_loss_max_db = float(combined_loss_db[idx_max])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta_div_urad, geometric_loss_db, label="Geometric loss (free space + gain TX)")
    ax.plot(theta_div_urad, pointing_loss_db, label="Pointing loss")
    ax.plot(theta_div_urad, combined_loss_db, label="Combined")
    ax.scatter(
        [theta_div_max_urad],
        [combined_loss_max_db],
        s=35,
        color="C2",
        zorder=6,
    )
    ax.annotate(
        f"{theta_div_max_urad:.2f} µrad",
        xy=(theta_div_max_urad, combined_loss_max_db),
        xytext=(10, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        color="C2",
        arrowprops=dict(arrowstyle="->", color="C2", lw=1),
    )
    ax.set_xlim(0.0, 40.0)
    ax.set_ylim(-50.0, 0.0)
    ax.set_xlabel("Divergence angle (µrad)")
    ax.set_ylabel("Loss (dB)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
