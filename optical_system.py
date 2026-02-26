optical_system1 = {
    "link_benchmark_specification": {
        altitude: 550,                  # km
        inclination: 70,                # degrees
        average_angle: 45,              # degrees
        average_contact_time: 30,       # minutes
        dataVolume: 2.5,                # Tb/day
    },
    "system_specifications": {
        system_frequency: 1550,                # nm
        system_minimum_temperature: -40,
        system_maximum_temperature: 40, # C

        platform_drift_angle: 0.1,     # degrees

        fsm_bandwidth: 1000, #Hz
        fsm_accuracy: 1*10^-6, #radians
        fsm_jitter: 1*10^-6, #radians

        transmitter_aperture: 0.1, #m
        receiver_aperture: 0.1, #m

        cpa_resolution: 500*10^-3, #radians
    },
    "other_specifications": {
        beam_width: 1,                  # degrees
        beam_jitter: 1,                 # degrees
        beam_divergence: 1,             # degrees
        beam_jitter_angle: 1,           # degrees
        beam_divergence_angle: 1,       # degrees
        beam_jitter_angle_std: 1,       # degrees
        beam_divergence_angle_std: 1,   # degrees
    },
    "noise_specifications" : {
        system_temperature: 300  # Kelvin
    }
}

