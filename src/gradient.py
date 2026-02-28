# src/gradient.py
import numpy as np

def calculate_gradient(rec_model, mesh_thicknesses, return_depth_centers=False):
    """
    Calculate vertical gradient of the recovered resistivity model.

    Parameters
    ----------
    rec_model : (nC,) array
        log-conductivity model (log(sigma)) recovered from inversion.
    mesh_thicknesses : (nC,) array
        Thicknesses for each layer (m).
    return_depth_centers : bool
        If True, also return cell-center depths (m bgs).

    Returns
    -------
    gradient : (nC,) array
        d/dz log(rho) evaluated per layer (cell centers).
    z_centers : (nC,) array, optional
        Cell-center depths (m bgs, positive down). Returned if return_depth_centers=True.
    """
    mesh_thicknesses = np.asarray(mesh_thicknesses, float).ravel()

    # Depth edges and centers
    depth_edges = np.r_[0.0, np.cumsum(mesh_thicknesses)]
    z_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])

    dz = np.diff(depth_edges)  # layer thicknesses

    # Convert log conductivity to resistivity
    model_res = 1.0 / np.exp(np.asarray(rec_model, float).ravel())
    logr = np.log(np.clip(model_res, 1e-12, None))

    gradient = np.zeros_like(model_res)

    # Central difference for interior layers (second-order, non-uniform grid)
    for i in range(1, len(model_res) - 1):
        h1, h2 = dz[i - 1], dz[i]
        gradient[i] = (
            h2**2 * logr[i - 1] +
            (h1**2 - h2**2) * logr[i] -
            h1**2 * logr[i + 1]
        ) / (h1 * h2 * (h1 + h2))

    # One-sided at the boundaries (keep your current scheme)
    gradient[0] = (-3 * logr[0] + 4 * logr[1] - logr[2]) / (2 * dz[0])
    gradient[-1] = (-3 * logr[-1] + 4 * logr[-2] - logr[-3]) / (2 * dz[-1])

    if return_depth_centers:
        return gradient, z_centers

    return gradient
