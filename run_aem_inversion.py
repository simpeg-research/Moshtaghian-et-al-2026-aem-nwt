# run_aem_inversion.py

import os
import pickle
import numpy as np
import pandas as pd

# Import custom inversion, mesh, survey, gradient, DOI, and plotting utilities
from src.inversion import (
    run_halfspace_inversion,
    run_multilayer_inversion_average_halfspace_initial,
    run_multilayer_inversion_average_halfspace_initial_fixed_beta0,
)
from src.survey import define_survey
from src.mesh import define_halfspace_mesh, define_multilayer_mesh
from src.gradient import calculate_gradient
from src.depth_of_investigation import (
    calculate_DOI_Oldenburg_Li_1999,
    calculate_DOI_Christiansen_Auken_2012,
)
from src.plotting import plot_resistivity_and_RMS


# -----------------------------
# Helpers: data extraction
# -----------------------------
def get_dobs(read_data, i, freqs_list):
    """Extract observed data as a flat float64 array (IP/Q interleaved)."""
    components = ["i", "q"]
    columns_data = [f"cp{comp}{freq}_prelim" for freq in freqs_list for comp in components]
    return read_data.iloc[i][columns_data].astype(np.float64).to_numpy()


# -----------------------------
# Helpers: PBTZ (center-based)
# -----------------------------
def _cell_centers_from_thicknesses(mesh_thicknesses):
    """Return depth centers (m, positive down) for a 1D layered mesh."""
    edges = np.r_[0.0, np.cumsum(np.asarray(mesh_thicknesses, float))]
    return 0.5 * (edges[:-1] + edges[1:])


def _win_local_from_dz(dz_eff, base=12.0, mult=3.0):
    """Same window idea as synthetic code: max(base, mult*dz)."""
    dz_eff = float(dz_eff)
    return max(float(base), float(mult) * dz_eff)


def _trapz_masked(z, y, mask):
    z = np.asarray(z, float)
    y = np.asarray(y, float)
    m = np.asarray(mask, bool)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(y[m], z[m]))


def _cmg_pbtz_from_center_gradient(
    zc,
    g_abs,
    z0,
    p=0.50,
    win_m=15.0,
    dz_floor=0.0,
    max_iter=30,
):
    """
    CMG-like TZ on centers:
    Find the narrowest connected interval around z0 containing fraction p of
    the local |d logρ / dz| "mass" within +/- win_m.

    Returns:
      z_top, z_bot, width, ok
    """
    zc = np.asarray(zc, float)
    g_abs = np.asarray(g_abs, float)

    mwin = (zc >= z0 - win_m) & (zc <= z0 + win_m)
    z = zc[mwin]
    gg = g_abs[mwin]

    if z.size < 3 or (not np.isfinite(gg).any()):
        return (z0 - dz_floor, z0 + dz_floor, 2.0 * dz_floor, False)

    # clean
    gg = np.where(np.isfinite(gg), gg, 0.0)
    gg = np.maximum(gg, 0.0)

    M = float(np.trapz(gg, z))
    if (not np.isfinite(M)) or (M <= 0.0):
        return (z0 - dz_floor, z0 + dz_floor, 2.0 * dz_floor, False)

    target = p * M
    gmax = float(np.max(gg))
    if (not np.isfinite(gmax)) or (gmax <= 0.0):
        return (z0 - dz_floor, z0 + dz_floor, 2.0 * dz_floor, False)

    # Binary search for threshold t_star such that area{gg>=t_star} ~= target
    lo, hi = 0.0, gmax
    for _ in range(max_iter):
        t = 0.5 * (lo + hi)
        area = _trapz_masked(z, gg, gg >= t)
        if area >= target:
            lo = t
        else:
            hi = t
        if abs(hi - lo) <= 1e-6 * gmax:
            break
    t_star = 0.5 * (lo + hi)

    mask = (gg >= t_star)
    if not mask.any():
        return (z0 - dz_floor, z0 + dz_floor, 2.0 * dz_floor, False)

    # Take the connected component containing z0 (closest index)
    i0 = int(np.argmin(np.abs(z - z0)))
    L = R = i0
    while (L - 1) >= 0 and mask[L - 1]:
        L -= 1
    while (R + 1) < mask.size and mask[R + 1]:
        R += 1

    # Interpolate edges at threshold between centers (linear in z)
    def interp_edge(i_left, i_right):
        z1, z2 = z[i_left], z[i_right]
        g1, g2 = gg[i_left], gg[i_right]
        if g2 == g1:
            return 0.5 * (z1 + z2)
        u = (t_star - g1) / (g2 - g1)
        u = np.clip(u, 0.0, 1.0)
        return float(z1 + u * (z2 - z1))

    z_top = float(z[L])
    z_bot = float(z[R])

    if L > 0 and (not mask[L - 1]):
        z_top = interp_edge(L - 1, L)
    if R < (mask.size - 1) and (not mask[R + 1]):
        z_bot = interp_edge(R, R + 1)

    # Enforce discretization floor
    width = max(z_bot - z_top, 2.0 * dz_floor)

    # Ensure interval contains z0 by shifting if needed
    if not (z_top <= z0 <= z_bot):
        mid = 0.5 * (z_top + z_bot)
        shift = z0 - mid
        z_top += shift
        z_bot += shift

    return (float(z_top), float(z_bot), float(width), True)


def _clamp_band_around_center(z0, width, dz_floor, w_min=None, w_max=None):
    """Clamp width with floors/priors and return (top, bot, width)."""
    width = max(float(width), float(dz_floor))
    if w_min is not None:
        width = max(width, float(w_min))
    if w_max is not None:
        width = min(width, float(w_max))
    return (float(z0 - 0.5 * width), float(z0 + 0.5 * width), float(width))


# -----------------------------
# Main workflow
# -----------------------------
def execute_inversion_workflow(
    selected_line,
    data_file="data/AEM_NWT_PaperLines.txt",
    output_dir="outputs",
    run_inversion=True,
    save_results=True,
    run_plotting=True,
    model_to_plot="fixedbeta",  # options: "fixedbeta", "avg"
    plot_doi_mode="none",       # options: "none", "oldenburg", "christiansen"
    line_name=None,
    # ---- NEW PBTZ config (repo-friendly defaults) ----
    compute_pbtz=True,
    pbtz_p=0.50,
    pbtz_full_width_min=2.0,    # m (optional prior)
    pbtz_full_width_max=10.0,   # m (optional prior)
):
    print(f"Running workflow for line: {selected_line}")

    # Load AEM data and extract lines
    df = pd.read_csv(data_file, delimiter="\t")
    line_names = df.iloc[:, 0].unique()
    if selected_line not in line_names:
        raise ValueError(f"Line {selected_line} not found in data!")

    # Filter data for the selected line
    read_data = df[df.iloc[:, 0] == selected_line]

    # System configuration parameters
    freqs_list = ["135k", "40k", "8200", "1800", "400"]
    frequencies = [135000, 40000, 8200, 1800, 400]
    coil_separations = [7.95, 7.93, 7.95, 7.94, 7.93]
    moment = np.array([17, 49, 72, 187, 359])
    source_orientation = receiver_orientation = "z"
    data_type = "ppm"

    # Extract coordinates and elevations
    x = np.array(read_data["x_tx"])
    y = np.array(read_data["y_tx"])
    dtm = np.array(read_data["dtm"])
    gpsz_tx = np.array(read_data["gpsz_tx"])
    b_height = gpsz_tx - dtm

    # Automatically choose horizontal coordinate for plotting
    source_loc = y if selected_line == "L120030" else x
    n_sounding = len(source_loc)

    # Define halfspace mesh and initial model
    sig_halfspace = 1e-2
    mesh_halfspace = define_halfspace_mesh([500])
    m0_hs = np.log(sig_halfspace) * np.ones(mesh_halfspace.nC)

    # Define multilayer mesh (used in both run and load paths)
    mesh, mesh_thicknesses = define_multilayer_mesh(1, 1.1, 41)
    zc = np.asarray(mesh.cell_centers_x, float)  # depth centers (+down)
    dz_eff = float(np.nanmedian(np.diff(np.r_[0.0, np.cumsum(mesh_thicknesses)])))
    win_local = _win_local_from_dz(dz_eff)
    dz_floor = 0.5 * dz_eff

    if run_inversion:
        # ---- Halfspace inversion to seed multilayer ----
        rec_hs_values = []
        for i in range(n_sounding):
            dobs = get_dobs(read_data, i, freqs_list)
            survey = define_survey(
                b_height[i], frequencies, coil_separations, moment,
                source_orientation, receiver_orientation, data_type
            )
            rec_model_hs, _ = run_halfspace_inversion(m0_hs, survey, mesh_halfspace, [500], dobs)
            rec_hs_values.append(rec_model_hs[0])

        # Create average starting model for multilayer inversions
        m0_hs_initial = float(np.mean(rec_hs_values))
        m0_avg = m0_hs_initial * np.ones(mesh.nC)

        # ---- Multilayer inversion helper ----
        def run_multilayer_loops(run_fn, is_fixed_beta=False, *args):
            results = {
                "model": [],
                "phi_d": [],
                "DOI_norm": [],
                "DOI_jac": [],
                "gradient": [],
                # ---- NEW ----
                "z0_depth": [],
                "pbtz_top_depth": [],
                "pbtz_bot_depth": [],
                "pbtz_width": [],
            }
            betas = {"avg": [], "avg10": [], "avg01": []} if not is_fixed_beta else {}

            for i in range(n_sounding):
                dobs = get_dobs(read_data, i, freqs_list)
                survey = define_survey(
                    b_height[i], frequencies, coil_separations, moment,
                    source_orientation, receiver_orientation, data_type
                )
                output = run_fn(m0_avg, survey, mesh, mesh_thicknesses, dobs, *args)

                # Parse inversion output
                if not is_fixed_beta:
                    (
                        ref_avg, model_avg, _, phi_d_avg, beta_avg, J_avg,
                        ref10, model10, _, _, beta10, J10,
                        ref01, model01, _, _, beta01, J01
                    ) = output
                    betas["avg"].append(beta_avg)
                    betas["avg10"].append(beta10)
                    betas["avg01"].append(beta01)
                    phi_d = phi_d_avg
                else:
                    (
                        ref_avg, model_avg, _, phi_d,
                        J_avg,
                        ref10, model10, _, _, J10,
                        ref01, model01, _, _, J01
                    ) = output

                # DOI metrics
                DOI_norm = calculate_DOI_Oldenburg_Li_1999(model10, model01, ref10, ref01)
                DOI_jac = calculate_DOI_Christiansen_Auken_2012(dobs, J_avg, mesh_thicknesses)

                # Gradient on centers (your existing function)
                grad = calculate_gradient(model_avg, mesh_thicknesses)  # length = n_layers (centers)

                # ---- NEW: center-based z0 + PBTZ ----
                grad_abs = np.abs(np.asarray(grad, float))
                if np.isfinite(grad_abs).any():
                    k = int(np.nanargmax(grad_abs))
                    z0_depth = float(zc[k])  # depth (+down)
                else:
                    z0_depth = np.nan

                if compute_pbtz and np.isfinite(z0_depth):
                    z_top_cmg, z_bot_cmg, w_cmg, ok = _cmg_pbtz_from_center_gradient(
                        zc=zc,
                        g_abs=grad_abs,
                        z0=z0_depth,
                        p=float(pbtz_p),
                        win_m=float(win_local),
                        dz_floor=float(dz_floor),
                    )
                    if not ok or (not np.isfinite(w_cmg)) or (w_cmg <= 0):
                        # fallback: one cell
                        z_top, z_bot, w_final = _clamp_band_around_center(
                            z0=z0_depth,
                            width=dz_eff,
                            dz_floor=dz_floor,
                            w_min=pbtz_full_width_min,
                            w_max=pbtz_full_width_max,
                        )
                    else:
                        # clamp with priors around z0 (keep it symmetric like your synthetic final band)
                        z_top, z_bot, w_final = _clamp_band_around_center(
                            z0=z0_depth,
                            width=w_cmg,
                            dz_floor=dz_floor,
                            w_min=pbtz_full_width_min,
                            w_max=pbtz_full_width_max,
                        )
                else:
                    z_top, z_bot, w_final = (np.nan, np.nan, np.nan)

                # Store outputs
                results["model"].append(model_avg)
                results["phi_d"].append(phi_d)
                results["DOI_norm"].append(DOI_norm)
                results["DOI_jac"].append(DOI_jac)
                results["gradient"].append(grad)

                results["z0_depth"].append([z0_depth])
                results["pbtz_top_depth"].append([z_top])
                results["pbtz_bot_depth"].append([z_bot])
                results["pbtz_width"].append([w_final])

            # Stack arrays
            results["model"] = np.vstack(results["model"])
            results["phi_d"] = np.asarray(results["phi_d"], float).reshape(-1, 1)
            results["DOI_norm"] = np.vstack(results["DOI_norm"])
            results["DOI_jac"] = np.asarray(results["DOI_jac"], float).reshape(-1, 1)
            results["gradient"] = np.vstack(results["gradient"])

            results["z0_depth"] = np.asarray(results["z0_depth"], float).reshape(-1)
            results["pbtz_top_depth"] = np.asarray(results["pbtz_top_depth"], float).reshape(-1)
            results["pbtz_bot_depth"] = np.asarray(results["pbtz_bot_depth"], float).reshape(-1)
            results["pbtz_width"] = np.asarray(results["pbtz_width"], float).reshape(-1)

            for k in betas:
                betas[k] = np.vstack(betas[k])

            return results, betas

        # ---- Dynamic-beta inversion ----
        results_avg, betas = run_multilayer_loops(run_multilayer_inversion_average_halfspace_initial)
        mean_betas = {k: float(np.mean(v)) for k, v in betas.items()}

        # ---- Fixed-beta inversion ----
        results_fixed_b0, _ = run_multilayer_loops(
            lambda m0, survey, mesh, mesh_thicknesses, dobs:
                run_multilayer_inversion_average_halfspace_initial_fixed_beta0(
                    m0,
                    mean_betas["avg"], mean_betas["avg10"], mean_betas["avg01"],
                    survey, mesh, mesh_thicknesses, dobs
                ),
            is_fixed_beta=True
        )

        # Save results to disk if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/{selected_line}_halfspace.pkl", "wb") as f:
                pickle.dump(rec_hs_values, f)
            with open(f"{output_dir}/{selected_line}_avg.pkl", "wb") as f:
                pickle.dump(results_avg, f)
            with open(f"{output_dir}/{selected_line}_fixedbeta.pkl", "wb") as f:
                pickle.dump(results_fixed_b0, f)

    else:
        # Load saved outputs if inversion not rerun
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{selected_line}_fixedbeta.pkl", "rb") as f:
            results_fixed_b0 = pickle.load(f)
        with open(f"{output_dir}/{selected_line}_avg.pkl", "rb") as f:
            results_avg = pickle.load(f)
        with open(f"{output_dir}/{selected_line}_halfspace.pkl", "rb") as f:
            rec_hs_values = pickle.load(f)

        # Backward compatibility: older pickles won't have PBTZ keys
        for res in (results_fixed_b0, results_avg):
            res.setdefault("z0_depth", np.full(n_sounding, np.nan))
            res.setdefault("pbtz_top_depth", np.full(n_sounding, np.nan))
            res.setdefault("pbtz_bot_depth", np.full(n_sounding, np.nan))
            res.setdefault("pbtz_width", np.full(n_sounding, np.nan))

    # -----------------------------
    # Plotting
    # -----------------------------
    if run_plotting:
        os.makedirs(output_dir, exist_ok=True)

        def _plot_one(results, label, out_png):
            # phi_d is stored as (n,1) above; plotting expects (n,) or (n,1) both OK
            plot_resistivity_and_RMS(
                source_loc,
                dtm,
                frequencies,
                mesh,
                mesh_thicknesses,
                model=results["model"],
                phi_d=results["phi_d"],
                doi_norm=results["DOI_norm"],
                doi_jac=results["DOI_jac"],
                gradient=results["gradient"],
                title=f"Line {selected_line} - {label}",
                filename=out_png,
                plot_doi_mode=plot_doi_mode,
                line_name=selected_line,
                # ---- NEW ----
                pbtz_top_depth=results.get("pbtz_top_depth", None),
                pbtz_bot_depth=results.get("pbtz_bot_depth", None),
            )

        if model_to_plot == "fixedbeta":
            _plot_one(
                results_fixed_b0,
                "Fixed β",
                f"{output_dir}/{selected_line}_fixedbeta_RMS.png"
            )
        elif model_to_plot == "avg":
            _plot_one(
                results_avg,
                "Dynamic β",
                f"{output_dir}/{selected_line}_avg_RMS.png"
            )
        else:
            raise ValueError("model_to_plot must be 'fixedbeta' or 'avg'.")

    print(f"Finished workflow for {selected_line}. Results in {output_dir}.")