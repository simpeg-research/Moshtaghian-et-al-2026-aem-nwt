# src/pbtz.py
import numpy as np

def _depth_edges_and_centers(mesh_thicknesses):
    """
    mesh_thicknesses: (nC,) layer thicknesses [m]
    returns:
      depth_edges:   (nC+1,) interface depths [m bgs]
      depth_centers: (nC,)   center depths [m bgs]
      dz_eff:        float   median layer thickness [m]
    """
    thk = np.asarray(mesh_thicknesses, float).ravel()
    depth_edges = np.r_[0.0, np.cumsum(thk)]
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    dz_eff = float(np.median(np.diff(depth_edges)))
    return depth_edges, depth_centers, dz_eff


def pick_z0_center(gradient, mesh_thicknesses, exclude_boundaries=True):
    """
    Pick z0 as argmax |gradient| evaluated at cell centers.

    gradient: (nC,) d/dz log(rho) at centers (your calculate_gradient output)
    returns z0 (m bgs)
    """
    _, zc, _ = _depth_edges_and_centers(mesh_thicknesses)
    gabs = np.abs(np.asarray(gradient, float)).copy()

    if exclude_boundaries and gabs.size >= 3:
        gabs[0] = np.nan
        gabs[-1] = np.nan

    if not np.isfinite(gabs).any():
        return np.nan

    i0 = int(np.nanargmax(gabs))
    return float(zc[i0])


def _trapz_masked(z, y, mask):
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    m = np.asarray(mask, bool)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(y[m], z[m]))


def calculate_pbtz_cmg(
    rec_model,
    mesh_thicknesses,
    z0=None,
    p=0.50,
    win_base=12.0,
    win_mult=3.0,
    width_min=2.0,
    width_max=10.0,
    dz_floor_factor=0.5,
    exclude_boundaries=True,
):
    """
    CMG-based PBTZ around z0 using |d log(rho)/dz| "mass" on centers.

    Inputs
    ------
    rec_model: (nC,) log-conductivity model (log(sigma))
    mesh_thicknesses: (nC,) layer thicknesses [m]
    z0: optional float (m bgs). If None, computed from max |gradient| at centers.
    p: fraction of gradient "mass" to capture (default 0.50)
    win_base, win_mult: adaptive half-window = max(win_base, win_mult*dz_eff)
    width_min, width_max: priors on full thickness [m]
    dz_floor_factor: discretization floor on full thickness = dz_floor_factor*dz_eff

    Returns
    -------
    z_top, z0, z_bot, width   (all in m bgs, positive down)
    """
    # geometry
    _, zc, dz_eff = _depth_edges_and_centers(mesh_thicknesses)

    # resistivity + log(rho)
    rho = 1.0 / np.exp(np.asarray(rec_model, float))
    logr = np.log(np.clip(rho, 1e-12, None))

    # center-based gradient mass
    g = np.abs(np.gradient(logr, zc, edge_order=2))

    # pick z0 (center) if not provided
    if z0 is None or not np.isfinite(z0):
        # z0 should come from the SAME gradient definition used in your repo
        # (call calculate_gradient outside and pass it in if you prefer)
        # Here we use g (from np.gradient) as a consistent fallback:
        gabs = g.copy()
        if exclude_boundaries and gabs.size >= 3:
            gabs[0] = np.nan
            gabs[-1] = np.nan
        if not np.isfinite(gabs).any():
            return np.nan, np.nan, np.nan, np.nan
        z0 = float(zc[int(np.nanargmax(gabs))])

    # window around z0
    win = max(float(win_base), float(win_mult) * dz_eff)
    mwin = (zc >= z0 - win) & (zc <= z0 + win)
    z, gg = zc[mwin], g[mwin]

    if z.size < 3 or not np.isfinite(gg).any():
        # fallback: minimal band around z0
        width = max(dz_floor_factor * dz_eff, width_min)
        width = min(width, width_max)
        return z0 - 0.5 * width, z0, z0 + 0.5 * width, width

    M = float(np.trapz(gg, z))
    if not np.isfinite(M) or M <= 0:
        width = max(dz_floor_factor * dz_eff, width_min)
        width = min(width, width_max)
        return z0 - 0.5 * width, z0, z0 + 0.5 * width, width

    target = p * M

    # threshold search (binary) for CMG band
    gmax = float(np.nanmax(gg))
    lo, hi = 0.0, gmax
    for _ in range(30):
        t = 0.5 * (lo + hi)
        area = _trapz_masked(z, gg, gg >= t)
        if area >= target:
            lo = t
        else:
            hi = t
    t_star = 0.5 * (lo + hi)

    mask = (gg >= t_star)
    if not mask.any():
        width = max(dz_floor_factor * dz_eff, width_min)
        width = min(width, width_max)
        return z0 - 0.5 * width, z0, z0 + 0.5 * width, width

    i0 = int(np.argmin(np.abs(z - z0)))
    L = R = i0
    while L - 1 >= 0 and mask[L - 1]:
        L -= 1
    while R + 1 < len(mask) and mask[R + 1]:
        R += 1

    z_top = float(z[L])
    z_bot = float(z[R])
    width = float(z_bot - z_top)

    # apply floors + priors, then recentre about z0
    width = max(width, dz_floor_factor * dz_eff)
    width = max(width, width_min)
    width = min(width, width_max)

    z_top = z0 - 0.5 * width
    z_bot = z0 + 0.5 * width

    return float(z_top), float(z0), float(z_bot), float(width)