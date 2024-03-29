module DrMZ

# Load required packages
using FFTW: fft, ifft
using Flux
using Flux.Data: DataLoader
using Flux: mse
using ProgressMeter: @showprogress
using Distributions: MvNormal
using LinearAlgebra
using Random: randperm, AbstractRNG
using Random
using DifferentialEquations
using Interpolations
using BSON: @save
using BSON: @load
using Printf
using Statistics: mean, std
using FastGaussQuadrature
using ForwardDiff
using LegendrePolynomials
using Kronecker


# Export the functions for General.jl
export error_se, error_rel, mse_error, norm_rel_error, norm_rel_error_continuous, norm_infinity_error, ic_error, average_ic_error, average_error,
       periodic_fill_domain, periodic_fill_solution, solution_interpolation, reduced_initial_condition, solution_spatial_sampling, solution_temporal_sampling,
       fft_norm, ifft_norm,
       trapz,
       gauss_legendre,
       clenshaw_curtis, cheby_grid, cheby_diff_matrix, cheby_diff,
       trapezoid,
       orthonormal_check,
       legendre_norm, legendre_norm_collect

# Export the functions for OperatorNN.jl
export train_model, loss_all, predict,
       build_dense_model, build_branch_model, build_trunk_model,
       exp_kernel_periodic, generate_sinusoidal_functions_2_parameter, generate_periodic_train_test_initial_conditions, generate_periodic_train_test,  generate_periodic_train_test_Adv2D,  generate_periodic_train_test_Diff2D, train_model_2D, loss_all_2D, generate_periodic_train_test_muscl, generate_periodic_train_test_esdirk, generate_periodic_train_test_implicit, generate_periodic_functions, solution_extraction,
       save_model, load_model, load_branch, load_trunk, save_data, load_data, load_data_initial_conditions, load_data_train_test, save_data_initial_conditions


# Export the functions for PDESolve.jl
export advection_pde!, advection_diffusion_pde!, inviscid_burgers_pde!, viscous_burgers_pde!, kdv_explicit_pde!, kdv_implicit_pde!, kdv_pde!, ks_explicit_pde!, ks_implicit_pde!, ks_pde!, rhs_advection!, rhs_advection_diffusion!, burgers_flux, quadratic_nonlinear_triple_product_basis, quadratic_nonlinear_basis, rhs_viscous_burgers!, rhs_inviscid_burgers!, rhs_explicit_kdv!, rhs_implicit_kdv!, rhs_kdv!, rhs_explicit_ks!, rhs_implicit_ks!, rhs_ks!,
       quadratic_nonlinear,
       generate_fourier_solution, generate_fourier_solution_esdirk, generate_fourier_solution_implicit, generate_basis_solution, generate_basis_solution_esdirk, generate_basis_solution_implicit,
       central_difference,
       minmod, ub, fl, ulpl, urpl, ulnl, urnl,
       muscl_minmod_RHS!, muscl_minmod_viscous_RHS!, generate_muscl_minmod_solution, generate_muscl_reduced,
       get_1D_energy_fft,
       get_1D_energy_custom, get_1D_energy_custom_coefficients,
       mode_extractor, get_1D_energy_upwind,
       spectral_approximation_fourier

# Export the functions for DBasis.jl
export trunk_build, trunk_ortho_build, build_basis,
       basis_eval, expansion_coefficients, expansion_approximation,
       basis_derivative, 
       save_basis, load_basis

# Load functions
include("General.jl")
include("OperatorNN.jl")
include("PDESolve.jl")
include("DBasis.jl")

end
