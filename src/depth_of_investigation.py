# src/depth_of_investigation.py

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
    model_10 = np.asarray(model_10, float)
    model_01 = np.asarray(model_01, float)
    reference_model_10 = np.asarray(reference_model_10, float)
    reference_model_01 = np.asarray(reference_model_01, float)

    denom = (reference_model_10 - reference_model_01)
    denom = np.where(np.abs(denom) > 0, denom, np.nan)

    DOI = (model_10 - model_01) / denom
    mx = np.nanmax(np.abs(DOI))
    if not np.isfinite(mx) or mx == 0:
        return np.full_like(DOI, np.nan, dtype=float)

    DOI_norm = DOI / mx
    return DOI_norm


def calculate_DOI_Christiansen_Auken_2012(dobs, J_dict, mesh_thicknesses, relative_error_value=0.05):
    """
    Calculate depth of investigation using Jacobian sensitivity approach.

    Reference:
    Christiansen, A. V., & Auken, E. (2012). A global measure for depth of investigation.
    Geophysics, 77(4), WB171–WB177.

    Parameters:
    - dobs: observed EM data (1D array length nD)
    - J_dict: Jacobian container. Accepts either:
        (i) dict with key 'ds'  -> J_dict['ds'] is (nD, nP)
        (ii) numpy array / sparse matrix directly -> treated as J
    - mesh_thicknesses: list/array of vertical layer thicknesses (length nC)
    - relative_error_value: assumed relative noise in the data (default = 5%)

    Returns:
    - DOI_jac: depth (positive, meters) above which cumulative normalized sensitivity exceeds 80%
    """
    # ---- Input cleanup ----
    dobs = np.asarray(dobs, float).ravel()
    thk = np.asarray(mesh_thicknesses, float).ravel()

    # ---- Extract Jacobian robustly (fixes your IndexError) ----
    if isinstance(J_dict, dict):
        if "ds" not in J_dict:
            raise KeyError(f"Jacobian dict must contain key 'ds'. Found: {list(J_dict.keys())}")
        J = J_dict["ds"]
    else:
        J = J_dict  # allow direct matrix/array

    # ---- Expected noise (avoid divide-by-zero if any dobs are ~0) ----
    delta_d = relative_error_value * np.abs(dobs)
    floor = np.nanmedian(delta_d[delta_d > 0]) if np.any(delta_d > 0) else 1.0
    delta_d = np.clip(delta_d, 0.1 * floor, None)

    # ---- Dimension-safe: match number of columns to number of layers ----
    # J is (nD, nP). Your layer count is len(thk)=nC.
    nC = thk.size
    nP = J.shape[1]
    n_use = min(nC, nP)

    J_use = J[:, :n_use]
    thk_use = thk[:n_use]

    # ---- Sensitivity per layer ----
    Sj = np.sum(np.abs(J_use) / delta_d[:, np.newaxis], axis=0)

    # Normalize by thickness (Christiansen & Auken concept)
    Sj_star = Sj / np.clip(thk_use, 1e-12, None)

    # Cumulative sensitivity from bottom to top
    cumulative_Sj = np.cumsum(Sj_star[::-1])[::-1]

    # Normalize cumulative sensitivity to [0,1] so 0.8 threshold makes sense
    c0 = cumulative_Sj[0] if cumulative_Sj.size > 0 else np.nan
    if not np.isfinite(c0) or c0 <= 0:
        return np.nan
    cumulative_norm = cumulative_Sj / c0

    # Thresholding: define layers with sufficient sensitivity
    active_doi = cumulative_norm >= 0.8
    if not np.any(active_doi):
        return np.nan

    # Depth to interfaces (positive down)
    depth_edges = np.r_[0.0, np.cumsum(thk_use)]

    # DOI depth = deepest interface that still satisfies the criterion
    last_idx = np.where(active_doi)[0][-1]
    DOI_jac = float(depth_edges[last_idx + 1])
    return DOI_jac