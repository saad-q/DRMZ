ENV["JULIA_CUDA_SILENT"] = true
# Change the working directory to the file location
cd(@__DIR__)

using DrMZ
using Parameters: @with_kw
using Flux.NNlib
using LaTeXStrings
using ColorSchemes
using Flux
using Plots; #pyplot()
using Plots.PlotMeasures
using Random
using Printf

default(fontfamily="serif",frame=:box,grid=:hide,palette=:viridis,markeralpha=0.4,dpi=200,legendfontsize=6);
#PyPlot.matplotlib.rc("mathtext",fontset="cm")

@with_kw mutable struct Args
    num_sensors_x::Int = 64;
    num_sensors_y::Int = 1;

    num_train_functions::Int = 500;
    num_test_functions::Int = Int(2*num_train_functions);
    num_sol_points::Int = 100;
    L_x = [0.0 1.0]';
    L_y = [0.0 0.0];
    M_x = 8;
    M_y = 0;
    alp = [1.0 0.0];

    tspan::Tuple = (0.0,1.0);
    n_epoch::Int = 7500;

    num_sensors = Int(num_sensors_x*num_sensors_y);

    N::Int = num_sensors;
    branch_layers::Int = 2; # Branch depth
    branch_neurons::Array{Tuple} = [(Int(num_sensors),num_sensors),(num_sensors,N)]; # Branch layers input and output dimensions - quantity of tuples must match branch depth
    branch_activations::Array = [tanh,identity]; # Activation functions for each branch layer
    trunk_layers::Int = 3; # Trunk depth
    trunk_neurons::Array{Tuple} = [(3,num_sensors),(num_sensors,num_sensors),(num_sensors,N)]; # Trunk layers input and output dimensions - quantity of tuples must match trunk depth
    trunk_activations::Array = [tanh,tanh,tanh]; # Activation functions for each trunk layer
end

function generate_train(pde_function;kws...)

    args = Args(;);

    # Generate training and testing data
    train_data, test_data = generate_periodic_train_test_Adv2D(args.L_x,args.L_y,args.M_x,args.M_y,args.tspan,args.alp,args.num_sensors_x,args.num_sensors_y,args.num_train_functions,args.num_test_functions,args.num_sol_points);
    save_data(train_data,test_data,args.num_train_functions,args.num_test_functions,args.num_sol_points,pde_function)

    branch = build_dense_model(args.branch_layers,args.branch_neurons,args.branch_activations);
    trunk = build_dense_model(args.trunk_layers,args.trunk_neurons,args.trunk_activations)

    # Train the operator neural network
    branch, trunk = train_model(branch,trunk,args.n_epoch,train_data,test_data,pde_function);
    save_model(branch,trunk,args.n_epoch,pde_function)

end


# Generate data and train the model
generate_train("advection2D_equation")


