"""
File to design lens and tracking detector.
"""

### DESIGN PARAMETERS ###
aperture = 0.2        # Effective system aperture (about half the FSM diameter) (m)
focal_length = 0.1     # Focal length of the lens (m)
angular_range = 1e-3   # FSM tracking range (rad)

### CONSTANT ###
wavelength = 1550e-9  # (m)

### CALCULATE ###
spot_size = 1.22 * wavelength * focal_length / aperture # Diffraction-limited spot size (m)
x_max = focal_length * angular_range                    # Max displacement of the spot on the detector (m)
sensitivity = x_max / spot_size                         # x_max should be >> spot size

print("=== PARAMS ===")
print(f"{aperture=} m")
print(f"{focal_length=} m")
print(f"{angular_range=} rad")
print("=== CALCULATED ===")
print(f"Spot size: {spot_size*1e6:.3f} µm")
print(f"Detector displacement: {x_max*1e3:.3f} mm")
print(f"x_max/w = {sensitivity:.3f}")
print(f"Minimum detector diameter: {3 * x_max*1e3:.3f} mm")