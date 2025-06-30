# src/plotting_utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_resistivity_and_RMS(source_loc, dtm, frequencies, mesh, mesh_thicknesses, model, phi_d,
                              doi_norm, doi_jac, gradient, title, filename,
                              plot_doi_mode="none", line_name=None):
    """
    Plot resistivity cross-section and RMS misfit.

    Parameters:
    - source_loc: x or y location of each sounding (based on line orientation)
    - dtm: ground elevation values (same length as source_loc)
    - frequencies: list of Tx frequencies used in inversion
    - mesh: 1D vertical mesh object
    - mesh_thicknesses: layer thicknesses used in inversion
    - model: log conductivity model (n_soundings x n_layers)
    - phi_d: data misfit values per sounding
    - doi_norm: normalized DOI from Oldenburg-Li method (n_soundings x n_layers)
    - doi_jac: DOI depth from Christiansen method (length n_soundings)
    - gradient: gradient-based permafrost base depths (n_soundings x n_layers)
    - title: plot title
    - filename: path to save the figure
    - plot_doi_mode: "none" | "oldenburg" | "christiansen"
    - line_name: line ID for applying gradient masking (e.g., "L120030")
    """
    # Convert model to resistivity and calculate RMS
    model_res = 1 / np.exp(model)
    RMS = np.sqrt(phi_d / (2 * len(frequencies)))

    # Calculate DOI depths (Oldenburg method)
    boundary_depths_DOI_fixed_b0 = []
    if plot_doi_mode == "oldenburg":
        for i in range(doi_norm.shape[0]):
            idx = np.where(doi_norm[i, :] <= 0.2)[0]
            boundary_depths_DOI_fixed_b0.append(mesh.cell_centers_x[idx[0]] if len(idx) > 0 else np.nan)

    # Setup mesh grid and topo-adjusted depth
    X, Z = np.meshgrid(source_loc, mesh.cell_centers_x)
    Z = -Z
    Z_topo = np.zeros_like(Z)
    DOI_topo_fixed_b0 = np.zeros(len(source_loc))
    DOI_jac_avg_topo_fixed_b0 = np.zeros(len(source_loc))
    permafrost_bottom_grad_avg_topo_fixed_b0 = np.zeros(len(source_loc))

    depths = np.insert(np.cumsum(mesh_thicknesses), 0, 0)
    for i in range(len(source_loc)):
        Z_topo[:, i] = Z[:, i] + dtm[i]
        if plot_doi_mode == "oldenburg":
            DOI_topo_fixed_b0[i] = -boundary_depths_DOI_fixed_b0[i] + dtm[i]
        if plot_doi_mode == "christiansen":
            DOI_jac_avg_topo_fixed_b0[i] = -doi_jac[i] + dtm[i]
        permafrost_bottom_grad_avg_topo_fixed_b0[i] = -depths[np.argmax(gradient[i, :])] + dtm[i]

    # Mask gradient line for specific lake-covered sections
    exclude_ranges = []
    if line_name == "L120030":
        exclude_ranges = [(7190850, 7191213), (7191668, 7191755), (7191880, 7192130), (7192595, 7192776),
                          (7193170, 7193207), (7193414, 7193436), (7193862, 7194050), (7194475, 7194665),
                          (7195868, 7196000), (7196394, 7196489), (7196749, 7196809), (7197280, 7199222),
                          (7200665, 7201013), (7201159, 7201649)]
    elif line_name == "L150020":
        exclude_ranges = [(406884, 406922), (407030, 407180)]
    elif line_name == "L160030":
        exclude_ranges = [(min(source_loc), 404260), (406207, 406290), (406783, 407004),
                          (407111, 407141), (407224, 407355), (407528, 407842),
                          (411640, 411683), (412056, 412132), (412280, max(source_loc))]

    combined_mask = np.zeros_like(source_loc, dtype=bool)
    for lo, hi in exclude_ranges:
        combined_mask |= (source_loc > lo) & (source_loc < hi)
    masked_grad = permafrost_bottom_grad_avg_topo_fixed_b0.copy()
    masked_grad[combined_mask] = np.nan

    # Fixed figure size and structured layout
    fig_width = 16
    fig_height = 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(14, 1, height_ratios=[2.0, 0.1, 0.2, 0.4, 1.5, 1.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05], hspace=0.0)

    # Plot resistivity model
    ax1 = fig.add_subplot(gs[0])
    contour1 = ax1.pcolormesh(X, Z_topo, np.log10(model_res.T), cmap='jet_r')
    contour1.set_clim(vmin=np.log10(1e0), vmax=np.log10(1e4))

    if plot_doi_mode == "oldenburg":
        masked_doi = np.where(doi_norm.T > 0.2, 1, np.nan)
        ax1.contourf(X, Z_topo, masked_doi, levels=[0.5, 1.5], colors='white', alpha=0.5)
        ax1.contour(X, Z_topo, doi_norm.T, levels=[0.2], colors='grey', linewidths=1.5, linestyles='dashed')
        ax1.plot([], [], linestyle='dashed', color='grey', label="DOI (Oldenburg)")
    elif plot_doi_mode == "christiansen":
        masked_doi_jac = np.where(Z_topo < DOI_jac_avg_topo_fixed_b0, 1, np.nan)
        ax1.contourf(X, Z_topo, masked_doi_jac, levels=[0.5, 1.5], colors='white', alpha=0.5)
        ax1.plot(source_loc, DOI_jac_avg_topo_fixed_b0, color='purple', linestyle='dashed', linewidth=1.5, label="DOI (Christiansen)")

    ax1.plot(source_loc, masked_grad, 'k-', linewidth=1.5, label="Gradient Base")
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=10)
    ax1.set_xlim(min(source_loc), max(source_loc))
    ax1.set_ylim(0, max(dtm))
    ax1.set_xticks([])
    ax1.set_aspect(aspect='auto')  # disables forced aspect
    ax1.legend(loc='upper right', fontsize=9, handlelength=2.0, framealpha=1.0, borderpad=0.4)

    # Colorbar right below resistivity plot
    ax_cb = fig.add_subplot(gs[3])
    cbar = fig.colorbar(contour1, cax=ax_cb, orientation='horizontal', format="$10^{%.1f}$")
    cbar.set_ticks(np.linspace(0, 4, 5))
    cbar.set_ticklabels(["1", "10", "100", "1000", "10000"])
    cbar.set_label(r"Resistivity ($\Omega$m)", labelpad=2)

    # RMS Misfit plot under colorbar
    ax2 = fig.add_subplot(gs[5], sharex=ax1)
    ax2.plot(source_loc, RMS, color="r", label="RMS")
    ax2.axhline(y=1, color='black', linestyle='dashed', linewidth=1.5)
    ax2.axhline(y=np.mean(RMS), color='red', linestyle='dashed', linewidth=1.5)
    ax2.set_ylabel("RMS Misfit", fontsize=10)
    ax2.set_title("RMS Misfit Across Source Locations", fontsize=12)
    ax2.legend(loc='upper right', fontsize=9, framealpha=1.0)
    ax2.set_ylim(0, 2)
    ax2.grid(True)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Horizontal Distance (m)', fontsize=10)

    # Save and show
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
