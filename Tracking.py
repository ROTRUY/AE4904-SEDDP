"""
File to design lens focal length and minimum tracking detector size.
"""
from math import pi
import sys
import numpy as np
import matplotlib.pyplot as plt

### DESIGN PARAMETERS ###
aperture =  0.1                            # Effective system aperture (m)
focal_length = 0.1
#focal_length = np.arange(0.01, 0.3, 0.001) # Focal length of the lens (m), can be float for single calculation, or array for graph
angular_range = 1                          # FSM tracking range
angular_range_unit = 'deg'                 # Unit of the angular range 'deg' or 'rad'

### CONSTANT ###
wavelength = 1550e-9  # (m)

### CALCULATE ###
if angular_range_unit == 'deg':
    angular_range_deg = angular_range
    angular_range_rad = angular_range * pi / 180
elif angular_range_unit == 'rad':
    angular_range_deg = angular_range * 180 / pi
    angular_range_rad = angular_range
else:
    print("INVALID ANGULAR RANGE UNIT")
    sys.exit()

spot_size = 1.22 * wavelength * focal_length / aperture # Diffraction-limited spot size (m)
x_max = focal_length * angular_range_rad                # Max displacement of the spot on the detector (m)
sensitivity = x_max / spot_size                         # x_max should be >> spot size

if type(focal_length) == float:  # If float, just print summary
    print("=== PARAMS ===")
    print(f"{aperture=} m")
    print(f"{focal_length=} m")
    print(f"{angular_range_rad=:.3f} rad")
    print(f"{angular_range_deg=:.3f} deg")
    print("=== CALCULATED ===")
    print(f"Spot size: {spot_size*1e6:.3f} µm")
    print(f"Detector displacement: {x_max*1e3:.3f} mm")
    print(f"x_max/w = {sensitivity:.3f}")
    print(f"Minimum detector diameter: ~{1.5 * x_max*1e3:.3f} mm")

else:  # If not a float, assumed array, plot all factors ifo focal length
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Spot size (µm)
    axs[0].plot(focal_length, spot_size * 1e6)
    axs[0].set_xlabel('Focal length (m)')
    axs[0].set_ylabel('Spot size (µm)')
    axs[0].set_title('Diffraction-limited spot size')
    axs[0].grid(True)

    # Detector displacement (mm)
    axs[1].plot(focal_length, x_max * 1e3)
    axs[1].set_xlabel('Focal length (m)')
    axs[1].set_ylabel('Detector displacement (mm)')
    axs[1].set_title('Max spot displacement on detector')
    axs[1].grid(True)

    plt.savefig("Plots\\focalLength.png", dpi=400)
