import numpy as np

# --------------------------
# SOLAR PANEL PARAMETERS
# --------------------------

solar_constant = 1361            # W/m^2
panel_efficiency = 0.28
system_losses = 0.77
panel_area = 4.0                 # m^2

sunlight_time = 60 * 60          # seconds per orbit

# Effective power density
power_density = solar_constant * panel_efficiency * system_losses

# Power generated
power_generated = power_density * panel_area

# --------------------------
# OPTICAL COMMUNICATIONS
# --------------------------

comms_power_fraction = 0.2               # W
comms_power = comms_power_fraction * power_generated      # seconds

print("Solar panel area:", panel_area, "m^2")

print("\nGenerated power:")
print(f"{power_generated:.1f} W")

print("\nComms power:")
print(f"{comms_power:.1f} W")

