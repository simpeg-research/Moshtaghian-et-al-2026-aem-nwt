# src/survey.py

import numpy as np
from simpeg.electromagnetics import frequency_domain as fdem


def define_survey(b_height, frequencies, coil_separations, moment, source_orientation, receiver_orientation, data_type):
    """
    Define a SimPEG frequency-domain EM survey for one sounding.

    Parameters:
    - b_height: height of the transmitter above the digital terrain model
    - frequencies: list of Tx frequencies
    - coil_separations: Tx-Rx horizontal coil spacings
    - moment: magnetic moments per frequency
    - source_orientation: orientation of the Tx coil ('x', 'y', or 'z')
    - receiver_orientation: orientation of Rx coil ('x', 'y', or 'z')
    - data_type: type of data to be used ('ppm', 'field')

    Returns:
    - A SimPEG fdem.Survey object representing all sources and receivers
    """

    source_list = []  # container for all magnetic dipole sources

    # Set the transmitter location (assumed centered at origin in X and Y)
    src_loc = np.c_[0.0, 0.0, b_height]  # shape: (1, 3)

    # Compute receiver offsets per frequency (assumed along X-axis)
    rx_offsets = np.vstack([np.r_[sep, 0.0, 0.0] for sep in coil_separations])

    for j in range(len(frequencies)):
        # Receiver location is offset from source
        rx_locs = src_loc - rx_offsets[j]

        # Define two Rx components: real and imaginary parts of magnetic field
        rx_real = fdem.receivers.PointMagneticFieldSecondary(
            rx_locs,
            orientation=receiver_orientation,
            data_type=data_type,
            component="real"
        )
        rx_imag = fdem.receivers.PointMagneticFieldSecondary(
            rx_locs,
            orientation=receiver_orientation,
            data_type=data_type,
            component="imag"
        )

        # Combine into a list of receivers for this frequency
        receivers_list = [rx_real, rx_imag]

        # Create a magnetic dipole source with its receivers
        src = fdem.sources.MagDipole(
            receivers_list,
            frequencies[j],
            src_loc,
            orientation=source_orientation,
            moment=moment[j]
        )

        # Append to survey list
        source_list.append(src)

    # Create and return full survey
    survey = fdem.Survey(source_list)
    return survey
