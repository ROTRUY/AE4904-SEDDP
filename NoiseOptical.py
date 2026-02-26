def noise(B: float | int, T_S: float | int):
    """
    Function to determine noise.

    :param B: Channel bandwidth [Hz]
    :type B: float | int
    :param T_S: System noise temperature [K]
    :type T_S: float | int 
    """
    k = 1.38e-23  # Boltzmann constant
    return k * T_S * B