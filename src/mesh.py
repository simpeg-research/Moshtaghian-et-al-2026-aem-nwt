# src/mesh.py

import numpy as np
import discretize


def define_halfspace_mesh(mesh_thicknesses_halfspace):
    """
    Define a 1D tensor mesh for halfspace inversion.

    Parameters:
    - mesh_thicknesses_halfspace: list containing thickness of the halfspace layer(s)

    Returns:
    - A discretize.TensorMesh object used for halfspace inversion
    """
    # Construct a single-layer mesh with one padding layer at the bottom
    mesh_halfspace = discretize.TensorMesh([
        np.r_[mesh_thicknesses_halfspace, mesh_thicknesses_halfspace[-1]]
    ], "0")

    return mesh_halfspace


def define_multilayer_mesh(first_layer_thickness, geometry_factor, n_layers):
    """
    Define a 1D tensor mesh with geometrically increasing thicknesses for multilayer inversion.

    Parameters:
    - first_layer_thickness: thickness of the topmost layer
    - geometry_factor: multiplication factor between successive layers (e.g., 1.1)
    - n_layers: total number of vertical layers (including final padding layer)

    Returns:
    - mesh: discretize.TensorMesh object for inversion and regularization
    - mesh_thicknesses: list of thicknesses for plotting or forward modeling
    """
    # Initialize list for thicknesses
    mesh_thicknesses = []

    # Compute geometric progression of thicknesses
    for i in range(n_layers - 1):
        thickness = first_layer_thickness * geometry_factor ** i
        mesh_thicknesses.append(thickness)

    # Add a final bottom padding layer with same thickness as last
    mesh = discretize.TensorMesh([
        np.r_[mesh_thicknesses, mesh_thicknesses[-1]]
    ], "0")

    return mesh, mesh_thicknesses
