import numpy as np

optical_system1 = {
    "link_benchmark_specs": {
        "altitude": 550*1000,             # m (satellite)
        "inclination": 70 * np.pi/180,    # radians (satellite)
        "elevation_angle": 45 * np.pi/180,  # radians (benchmark - avg of link per 24hrs)
        "average_contact_time": 30,       # minutes (average contact time per 24hrs)
        "dataVolume": 2.5 * 1e12,        # bits/day (required data volume per day)
    },
    "transmitter_specs": {
        "system_frequency": 1550*1e-9,        # m
        "system_minimum_temperature": -40,     # C
        "system_maximum_temperature": 40,      # C
        "platform_drift_angle": 0.1,     # degrees
        "fsm_bandwidth": 1000, #Hz
        "fsm_accuracy": 1e-6, #radians
        "fsm_jitter": 1e-6, #radians
        "transmitter_aperture": 0.1, #m
        "transmitter_divergence_angle": 20*1e-6, #radians (taken from slides)
        "transmitter_laser_power": 40, #dBm (10W = 30dBm)
        "transmitter_static_pointing_error": 10*1e-6, #radians  # TODO define this further based on fsm specs or otherwise
        "transmission_optics": 0.5, # from slides 
    },
    "receiver_specs": {
        "receiver_aperture": 0.405, #m  (assuming R16 telescope Delft optical diameter)
        "receiver_outage_power": 0.2, # normalised (1 = max received power) (taken from slides)
        "outage_probability": 1e-3, # suggested assumption (0.1% outage)
        "receiver_avg_power": 0.85, # normalised (taken from slides)
        "cpa_resolution": 500*1e-3, #radians
    },
    "other_specs": {
        "beam_width": 1,                 # degrees
        "beam_jitter": 1,                 # degrees
        "beam_divergence": 1,             # degrees
        "beam_jitter_angle": 1,           # degrees
        "beam_divergence_angle": 1,       # degrees
        "beam_jitter_angle_std": 1,       # degrees
        "beam_divergence_angle_std": 1,   # degrees
    },
    "noise_specifications" : {
        "system_temperature": 300  # Kelvin
    }
}

