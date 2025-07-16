# src/gradient.py

import numpy as np

def calculate_gradient(rec_model, mesh_thicknesses):
    """
    Calculate vertical gradient of the recovered resistivity model using second-order finite differences.

    Parameters:
    - rec_model: log-conductivity model (np.log(sigma)) recovered from inversion
    - mesh_thicknesses: list of thicknesses for each vertical layer

    Returns:
    - gradient: vertical gradient of log-resistivity (∂log(ρ)/∂z) at each depth layer
    """
    # Compute cumulative depth and spacing
    depths = np.cumsum(mesh_thicknesses)
    depths = np.insert(depths, 0, 0)  # prepend surface depth = 0
    dz = np.diff(depths)              # layer thicknesses

    # Convert log conductivity to resistivity
    model_res = 1 / np.exp(rec_model)

    # Initialize gradient array
    gradient = np.zeros_like(model_res)

    # Central difference for interior layers (second-order accurate)
    for i in range(1, len(model_res) - 1):
        h1, h2 = dz[i - 1], dz[i]
        gradient[i] = (
            h2**2 * np.log(model_res[i - 1]) +
            (h1**2 - h2**2) * np.log(model_res[i]) -
            h1**2 * np.log(model_res[i + 1])
        ) / (h1 * h2 * (h1 + h2))

    # Forward difference at the surface
    gradient[0] = (
        -3 * np.log(model_res[0]) +
        4 * np.log(model_res[1]) -
        np.log(model_res[2])
    ) / (2 * dz[0])

    # Backward difference at the bottom
    gradient[-1] = (
        -3 * np.log(model_res[-1]) +
        4 * np.log(model_res[-2]) -
        np.log(model_res[-3])
    ) / (2 * dz[-1])

    return gradient
