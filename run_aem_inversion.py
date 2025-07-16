# run_aem_inversion.py

import pandas as pd
import numpy as np
import os
import pickle

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

# Helper function to extract observed data as a flat float64 array
def get_dobs(read_data, i, freqs_list):
    components = ['i', 'q']
    columns_data = [f"cp{comp}{freq}_prelim" for freq in freqs_list for comp in components]
    return read_data.iloc[i][columns_data].astype(np.float64).to_numpy()

def execute_inversion_workflow(
    selected_line,
    data_file='data/AEM_NWT_PaperLines.txt',
    output_dir='outputs',
    run_inversion=True,
    save_results=True,
    run_plotting=True,
    model_to_plot="fixedbeta",  # options: "fixedbeta", "avg"
    plot_doi_mode="none", # Options: "none", "oldenburg", "christiansen"
    line_name=None
):
    print(f"Running workflow for line: {selected_line}")

    # Load AEM data and extract lines
    df = pd.read_csv(data_file, delimiter='\t')
    line_names = df.iloc[:, 0].unique()
    if selected_line not in line_names:
        raise ValueError(f"Line {selected_line} not found in data!")

    # Filter data for the selected line
    read_data = df[df.iloc[:, 0] == selected_line]

    # System configuration parameters
    freqs_list = ['135k', '40k', '8200', '1800', '400']
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

    if run_inversion:
        # Run halfspace inversion and collect initial resistivity values
        rec_hs_values = []
        for i in range(n_sounding):
            dobs = get_dobs(read_data, i, freqs_list)
            survey = define_survey(b_height[i], frequencies, coil_separations, moment, source_orientation, receiver_orientation, data_type)
            rec_model_hs, _ = run_halfspace_inversion(m0_hs, survey, mesh_halfspace, [500], dobs)
            rec_hs_values.append(rec_model_hs[0])

        # Create average starting model for multilayer inversions
        m0_hs_initial = np.mean(rec_hs_values)
        mesh, mesh_thicknesses = define_multilayer_mesh(1, 1.1, 41)
        m0_avg = m0_hs_initial * np.ones(mesh.nC)

        # Multilayer inversion helper function
        def run_multilayer_loops(run_fn, is_fixed_beta=False, *args):
            results = {'model': [], 'phi_d': [], 'DOI_norm': [], 'DOI_jac': [], 'gradient': []}
            betas = {'avg': [], 'avg10': [], 'avg01': []} if not is_fixed_beta else {}

            for i in range(n_sounding):
                dobs = get_dobs(read_data, i, freqs_list)
                survey = define_survey(b_height[i], frequencies, coil_separations, moment, source_orientation, receiver_orientation, data_type)
                output = run_fn(m0_avg, survey, mesh, mesh_thicknesses, dobs, *args)

                # Parse inversion output
                if not is_fixed_beta:
                    ref_avg, model_avg, _, _, beta_avg, J_avg, ref10, model10, _, _, beta10, J10, ref01, model01, _, _, beta01, J01 = output
                    betas['avg'].append(beta_avg)
                    betas['avg10'].append(beta10)
                    betas['avg01'].append(beta01)
                else:
                    ref_avg, model_avg, _, phi_d, J_avg, ref10, model10, _, _, J10, ref01, model01, _, _, J01 = output

                # Compute DOI and gradients
                DOI_norm = calculate_DOI_Oldenburg_Li_1999(model10, model01, ref10, ref01)
                DOI_jac = calculate_DOI_Christiansen_Auken_2012(dobs, J_avg, mesh_thicknesses)
                grad = calculate_gradient(model_avg, mesh_thicknesses)

                # Store all outputs
                results['model'].append(model_avg)
                results['phi_d'].append(phi_d if is_fixed_beta else _)
                results['DOI_norm'].append(DOI_norm)
                results['DOI_jac'].append(DOI_jac)
                results['gradient'].append(grad)

            for k in results:
                results[k] = np.vstack(results[k])
            for k in betas:
                betas[k] = np.vstack(betas[k])

            return results, betas

        # Run dynamic-beta inversion
        results_avg, betas = run_multilayer_loops(run_multilayer_inversion_average_halfspace_initial)
        mean_betas = {k: np.mean(v) for k, v in betas.items()}

        # Run fixed-beta inversion
        results_fixed_b0, _ = run_multilayer_loops(
            lambda m0, survey, mesh, mesh_thicknesses, dobs:
                run_multilayer_inversion_average_halfspace_initial_fixed_beta0(
                    m0, mean_betas['avg'], mean_betas['avg10'], mean_betas['avg01'], survey, mesh, mesh_thicknesses, dobs
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
        mesh, mesh_thicknesses = define_multilayer_mesh(1, 1.1, 41)
        with open(f"{output_dir}/{selected_line}_fixedbeta.pkl", "rb") as f:
            results_fixed_b0 = pickle.load(f)
        with open(f"{output_dir}/{selected_line}_avg.pkl", "rb") as f:
            results_avg = pickle.load(f)
        with open(f"{output_dir}/{selected_line}_halfspace.pkl", "rb") as f:
            rec_hs_values = pickle.load(f)

    if run_plotting:
        # Create output directory and generate resistivity/RMS plots
        os.makedirs(output_dir, exist_ok=True)

        if model_to_plot == "fixedbeta":
            plot_resistivity_and_RMS(
                source_loc,
                dtm,
                frequencies,
                mesh,
                mesh_thicknesses,
                model=results_fixed_b0['model'],
                phi_d=results_fixed_b0['phi_d'],
                doi_norm=results_fixed_b0['DOI_norm'],
                doi_jac=results_fixed_b0['DOI_jac'],
                gradient=results_fixed_b0['gradient'],
                title=f"Line {selected_line} - Fixed β",
                filename=f"{output_dir}/{selected_line}_fixedbeta_RMS.png",
                plot_doi_mode=plot_doi_mode,
                line_name=selected_line
            )

        elif model_to_plot == "avg":
            plot_resistivity_and_RMS(
                source_loc,
                dtm,
                frequencies,
                mesh,
                mesh_thicknesses,
                model=results_avg['model'],
                phi_d=results_avg['phi_d'],
                doi_norm=results_avg['DOI_norm'],
                doi_jac=results_avg['DOI_jac'],
                gradient=results_avg['gradient'],
                title=f"Line {selected_line} - Dynamic β",
                filename=f"{output_dir}/{selected_line}_avg_RMS.png",
                plot_doi_mode=plot_doi_mode,
                line_name=selected_line
            )

    print(f"Finished workflow for {selected_line}. Results in {output_dir}.")
