# src/DOI_utils.py

import numpy as np

def calculate_DOI_Oldenburg_Li_1999(model_10, model_01, reference_model_10, reference_model_01):
    """
    Calculate depth of investigation (DOI) using the reference model difference approach.

    Reference:
    Oldenburg, D. W., & Li, Y. (1999). Estimating depth of investigation in DC resistivity and IP surveys.
    Geophysics, 64(2), 403–416.

    Parameters:
    - model_10: recovered model using average*10 as the reference
    - model_01: recovered model using average/10 as the reference
    - reference_model_10: log conductivity prior of average*10
    - reference_model_01: log conductivity prior of average/10

    Returns:
    - DOI_norm: normalized DOI indicator (0 to 1)
    """
    DOI = (model_10 - model_01) / (reference_model_10 - reference_model_01)
    DOI_norm = DOI / np.max(DOI)
    return DOI_norm


def calculate_DOI_Christiansen_Auken_2012(dobs, J_dict, mesh_thicknesses, relative_error_value=0.05):
    """
    Calculate depth of investigation using Jacobian sensitivity approach.

    Reference:
    Christiansen, A. V., & Auken, E. (2012). A global measure for depth of investigation.
    Geophysics, 77(4), WB171–WB177.

    Parameters:
    - dobs: observed EM data
    - J_dict: dictionary containing the Jacobian matrix from simulation (key: 'ds')
    - mesh_thicknesses: list of vertical layer thicknesses
    - relative_error_value: assumed relative noise in the data (default = 5%)

    Returns:
    - DOI_jac: depth (positive) above which sensitivity exceeds 80% threshold
    """
    delta_d = relative_error_value * np.abs(dobs)  # expected noise level
    J = J_dict['ds']  # extract sensitivity matrix (data vs. model)

    # Sensitivity summation across all data points for each layer
    Sj = np.sum(np.abs(J[:, :-1]) / delta_d[:, np.newaxis], axis=0)

    # Normalize by thickness (as per Eqn 4 in Christiansen & Auken)
    Sj_star = Sj / mesh_thicknesses

    # Cumulative sensitivity from bottom to top (Eqn 5)
    cumulative_Sj = np.cumsum(Sj_star[::-1])[::-1]

    # Thresholding: define layers with sufficient sensitivity
    active_doi = cumulative_Sj - 0.8 > 0.0

    # Build depth vector (negative downwards)
    depth = np.r_[0.0, -np.cumsum(mesh_thicknesses[:-1])]

    # Maximum depth above which the model is considered reliable
    DOI_jac = abs(depth[active_doi]).max()
    return DOI_jac
