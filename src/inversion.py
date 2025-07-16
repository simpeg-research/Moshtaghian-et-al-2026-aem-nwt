# src/inversion.py

import numpy as np
from simpeg import (
    maps,
    optimization,
    data_misfit,
    regularization,
    inverse_problem,
    inversion,
    directives,
    data
)
from simpeg.electromagnetics import frequency_domain as FDEM

# -------------------------------------------------------------------------
# Halfspace inversion for one sounding (single-layer model)
# -------------------------------------------------------------------------
def run_halfspace_inversion(m0_hs, survey, mesh_halfspace, mesh_thicknesses_halfspace, dobs):
    """
    Perform 1D halfspace inversion using a single conductivity value.
    """
    log_conductivity_map_hs = maps.ExpMap()

    # Define forward model
    prob_hs = FDEM.Simulation1DLayered(
        survey=survey,
        thicknesses=mesh_thicknesses_halfspace,
        sigmaMap=log_conductivity_map_hs
    )

    # Define data and uncertainty model
    uncertainties = 0.05 * np.abs(dobs) * np.ones(np.shape(dobs))
    dat_hs = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

    # Data misfit
    dmisfit_hs = data_misfit.L2DataMisfit(simulation=prob_hs, data=dat_hs)

    # Regularization (identity map for simple 1-parameter model)
    regMesh_hs = maps.IdentityMap(nP=mesh_halfspace.nC)
    reg_sigma_hs = regularization.WeightedLeastSquares(mesh_halfspace, mapping=regMesh_hs, alpha_s=1e-4)

    # Optimization setup
    opt_hs = optimization.ProjectedGNCG(maxIter=30, maxIterLS=20, maxIterCG=30, tolCG=1e-3)

    # Inversion problem setup
    inv_prob_hs = inverse_problem.BaseInvProblem(dmisfit_hs, reg_sigma_hs, opt_hs)

    # Inversion directives
    starting_beta_hs = directives.BetaEstimate_ByEig(beta0_ratio=5)
    beta_schedule_hs = directives.BetaSchedule(coolingFactor=1.5, coolingRate=2)
    target_misfit_hs = directives.TargetMisfit(chifact=1)

    directives_list_hs = [starting_beta_hs, beta_schedule_hs, target_misfit_hs]

    # Run inversion
    inv_hs = inversion.BaseInversion(inv_prob_hs, directives_list_hs)
    rec_model_hs = inv_hs.run(m0_hs)
    phi_d_final_hs = dmisfit_hs(rec_model_hs)

    return rec_model_hs, phi_d_final_hs

# -------------------------------------------------------------------------
# Multilayer inversion with average halfspace initialization
# Run 3 inversions with initial model, model*10, model/10 to support DOI
# -------------------------------------------------------------------------
def run_multilayer_inversion_average_halfspace_initial(m0_avg, survey, mesh, mesh_thicknesses, dobs):
    """
    Run 3 multilayer inversions with average, average*10, average/10 initializations.
    Used for DOI estimation and sensitivity analysis.
    """
    reference_model_avg = m0_avg
    reference_model_avg_10 = m0_avg - np.log(10)
    reference_model_avg_01 = m0_avg + np.log(10)

    # Perform each inversion separately
    rec_model_avg, dpred_avg, phi_d_avg, bet0_avg, J_dict_avg = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg)
    rec_model_avg_10, dpred_avg_10, phi_d_avg_10, bet0_avg_10, J_dict_avg_10 = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg_10)
    rec_model_avg_01, dpred_avg_01, phi_d_avg_01, bet0_avg_01, J_dict_avg_01 = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg_01)

    return reference_model_avg, rec_model_avg, dpred_avg, phi_d_avg, bet0_avg, J_dict_avg, \
           reference_model_avg_10, rec_model_avg_10, dpred_avg_10, phi_d_avg_10, bet0_avg_10, J_dict_avg_10, \
           reference_model_avg_01, rec_model_avg_01, dpred_avg_01, phi_d_avg_01, bet0_avg_01, J_dict_avg_01

# -------------------------------------------------------------------------
# Multilayer inversion with average halfspace initialization and fixed beta
# Used to stabilize inversion and keep regularization consistent
# -------------------------------------------------------------------------
def run_multilayer_inversion_average_halfspace_initial_fixed_beta0(m0_avg, mean_bet0_avg_append, mean_bet0_avg_10_append, mean_bet0_avg_01_append, survey, mesh, mesh_thicknesses, dobs):
    """
    Run multilayer inversion for one sounding with fixed beta0 values from prior runs.
    """
    reference_model_avg = m0_avg
    reference_model_avg_10 = m0_avg - np.log(10)
    reference_model_avg_01 = m0_avg + np.log(10)

    rec_model_avg, dpred_avg, phi_d_avg, _, J_dict_avg = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg, beta_fixed=mean_bet0_avg_append)
    rec_model_avg_10, dpred_avg_10, phi_d_avg_10, _, J_dict_avg_10 = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg_10, beta_fixed=mean_bet0_avg_10_append)
    rec_model_avg_01, dpred_avg_01, phi_d_avg_01, _, J_dict_avg_01 = _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model_avg_01, beta_fixed=mean_bet0_avg_01_append)

    return reference_model_avg, rec_model_avg, dpred_avg, phi_d_avg, J_dict_avg, \
           reference_model_avg_10, rec_model_avg_10, dpred_avg_10, phi_d_avg_10, J_dict_avg_10, \
           reference_model_avg_01, rec_model_avg_01, dpred_avg_01, phi_d_avg_01, J_dict_avg_01

# -------------------------------------------------------------------------
# Internal helper: Perform 1 multilayer inversion with optional fixed beta
# -------------------------------------------------------------------------
def _run_single_multilayer(survey, mesh, mesh_thicknesses, dobs, reference_model, beta_fixed=None):
    """
    Internal helper function to perform multilayer inversion with or without fixed beta.
    """
    conductivity_map = maps.ExpMap()
    sim = FDEM.Simulation1DLayered(
        survey=survey,
        thicknesses=mesh_thicknesses,
        sigmaMap=conductivity_map
    )

    uncertainties = 0.05 * np.abs(dobs) * np.ones(np.shape(dobs))
    dat = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

    dmisfit = data_misfit.L2DataMisfit(simulation=sim, data=dat)

    reg_map = maps.IdentityMap(nP=mesh.nC)
    reg = regularization.WeightedLeastSquares(mesh, mapping=reg_map, reference_model=reference_model, alpha_s=1e-4)

    opt = optimization.ProjectedGNCG(maxIter=30, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
    inv_prob = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

    if beta_fixed is not None:
        inv_prob.beta = beta_fixed
        directive_list = [directives.TargetMisfit(chifact=1)]
    else:
        directive_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=5),
            directives.BetaSchedule(coolingFactor=1.5, coolingRate=2),
            directives.TargetMisfit(chifact=1)
        ]

    inv = inversion.BaseInversion(inv_prob, directive_list)
    rec_model = inv.run(reference_model)
    phi_d_final = dmisfit(rec_model)
    dpred = inv_prob.dpred
    J_dict = sim.getJ(rec_model)

    return rec_model, dpred, phi_d_final, inv_prob.beta, J_dict
