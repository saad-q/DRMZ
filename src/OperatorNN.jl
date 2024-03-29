"""
    predict(branch,trunk,initial_condition,x_locations,t_values)
Predict solution ``u(t,x)`` at specified output locations using trained operator neural network `branch` and `trunk`.
"""
function predict(branch,trunk,initial_condition,x_locations,t_values)
    u = zeros(size(t_values,1),size(x_locations,1));
    bk = branch(initial_condition)';
    for i in 1:size(t_values,1)
        t_mat = repeat([t_values[i]],1,size(x_locations,1));
        t_x_mat = vcat(t_mat,x_locations');
        u[i,:] = bk*trunk(t_x_mat);
    end

    return u
end

"""
    loss_all(branch,trunk,initial_conditon,solution_location,target_value)
Compute the mean squared error (MSE) for a complete dataset.
"""
function loss_all(branch,trunk,initial_condition,solution_location,target_value)
    yhat = zeros(1,size(target_value,2));
    for i in 1:size(target_value,2)
        yhat[i] = branch(initial_condition[:,i])'*trunk(solution_location[:,i]);
    end
    return (1/size(target_value,2))*sum((yhat.-target_value).^2);
end

"""
    function build_dense_model(number_layers,neurons,activations)
Build a feedforward neural network (FFNN) consisting of `number_layers` of Flux dense layers for the specified number of `neurons` and `activations`.
# Examples
```julia-repl
julia> build_dense_model(2,[(128,128),(128,128)],[relu,relu])
Chain(Dense(128, 128, NNlib.relu), Dense(128, 128, NNlib.relu))
```
"""
function build_dense_model(number_layers,neurons,activations)
    layers = [Dense(neurons[i][1],neurons[i][2],activations[i]) for i in 1:number_layers];
    return Chain(layers...)|>f64
end

"""
    train_model(branch,trunk,n_epoch,train_data;learning_rate=0.00001,save_at=2500,starting_epoch=0)
Train the operator neural network using the mean squared error (MSE) and Adam optimization for `n_epochs` epochs.
"""
function train_model(branch,trunk,n_epoch,train_data,test_data,pde_function;learning_rate=1e-5,save_at=2500,starting_epoch=0)
    loss(x,y,z) = Flux.mse(branch(x)'*trunk(y),z)
    par = Flux.params(branch,trunk);
    opt = ADAM(learning_rate);
    @showprogress 1 "Training the model..." for i in 1:n_epoch
        Flux.train!(loss,par,train_data,opt);
        if i%save_at == 0
            save_model(branch,trunk,Int(starting_epoch+i),(pde_function*"_temp"))
            train_MSE = loss_all(branch,trunk,train_data.data[1],train_data.data[2],train_data.data[3]);
            test_MSE = loss_all(branch,trunk,test_data.data[1],test_data.data[2],test_data.data[3]);
            println("Train MSE $train_MSE")
            println("Test MSE $test_MSE")
        end
    end
    return branch,trunk
end

"""
    function exp_kernel_periodic(fnc,x_locations;length_scale=0.5)
Covariance kernel for radial basis function (GRF) and IC function for domain.
"""
function exp_kernel_periodic(fnc,x_locations,length_scale)
    out = zeros(size(x_locations,1),size(x_locations,1));
    out = [-(1/(2*length_scale^2))*norm(fnc(x_locations[i]) - fnc(x_locations[j]))^2 for i in 1:size(x_locations,1), j in 1:size(x_locations,1)]
    return exp.(out)
end

"""
    generate_periodic_functions(fnc,x_locations,number_functions,length_scale)
Generate a specified `number_functions` of random periodic vectors using the `exp_kernel_periodic` function and a multivariate distribution.
"""
function generate_periodic_functions(fnc,x_locations,number_functions,length_scale)
    sigma = exp_kernel_periodic(fnc,x_locations,length_scale); # Covariance
    mu = zeros(size(x_locations,1)); # Zero mean
    # Force the covariance matrix to be positive definite, by construction it is "approximately" Hermitian but Julia is strict
    mineig = eigmin(sigma);
    sigma -= mineig * I;
    d = MvNormal(mu,Symmetric(sigma)); # Multivariate distribution
    return rand(d,Int(number_functions))
end

"""
    generate_sinusoidal_functions_2_parameter(x_locations,number_functions)
Generate a specified `number_functions` of random periodic vectors for the distribution \$α \\sin(x)+β\$ for ``α ∈ [-1,1]`` and ``β ∈ [-1,1]``.
"""
function generate_sinusoidal_functions_2_parameter(x_locations,number_functions)
    values = rand(-1.0:(2.0/(number_functions*10)):1.0,(number_functions,2)); # alpha, beta parameters
    initial_base = vcat(values,[0.0 0.0]);
    if size(unique(initial_base,dims=2),1) != size(initial_base,1)
        error("Duplicate multiplier sets or generating function included in data!")
    end
    initial_conditions = zeros(size(x_locations,1),number_functions);
    for i in 1:number_functions
        initial_conditions[:,i] = values[i,1]*sin.(x_locations).+values[i,2]; # αsin(x)+β
    end
    return initial_conditions
end

"""
    solution_extraction(x_locations,t_values,solution,initial_condition,number_solution_points)
Extract the specified `number_solution_points` randomly from the ``u(t,x)`` solution space.
"""
function solution_extraction(x_locations,t_values,solution,initial_condition,number_solution_points)
    if number_solution_points >= size(solution,1)*size(solution,2)
        error("Invalid number of test points for the given dataset size!")
    else
        # Create a grid of the t,x data in the form of an array of tuples for accessing below
        t_x_grid = Array{Tuple{Float64,Float64}}(undef,(size(solution,1), size(solution,2)));
        for i in 1:size(solution,1)
            for j in 1:size(solution,2)
                t_x_grid[i,j] = (t_values[i],x_locations[j]);
            end
        end
        lin_ind = reduce(vcat,size(solution[:]));
        shuffled_indices = randperm(lin_ind) # Randomly shuffle the linear indices corresponding to the t,x data
        indices = shuffled_indices[1:number_solution_points] # From shuffled data, extract the number of test points for training
        initial = repeat(reshape(initial_condition,(length(initial_condition),1)),1,number_solution_points);
        sol_location = zeros(2,number_solution_points);
        for i in 1:2
            for j in 1:number_solution_points
                sol_location[i,j] = t_x_grid[indices[j]][i]
            end
        end
        sol = reshape(solution[indices],(1,number_solution_points));
        return initial, sol_location, sol
    end
end

"""
    generate_periodic_train_test_initial_conditions(L1,L2,number_sensors,number_test_functions,number_train_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt=1e-3,nu_val=0.1,domain="periodic",fnc=(x)->sin(x/2)^2)
Generate the training and testing data for a specified `pde_function_handle` for periodic boundary conditions using a Fourier spectral method. Defaults to IC \$f(\\sin^2(x/2))\$ and \$x ∈ [0,1]\$.
"""
function generate_periodic_train_test_initial_conditions(L1,L2,number_sensors,number_train_functions,number_test_functions;length_scale=0.5,domain="periodic",fnc=(x)->sin(x/2)^2)

    if domain == "periodic"
        x_full = range(L1,stop = L2,length = number_sensors+1); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale);
    elseif domain == "full"
        x_full = range(L1,stop = L2,length = number_sensors); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale);
    end

    train_ic = random_ics[:,1:number_train_functions];
    test_ic = random_ics[:,number_train_functions+1:end];

    return train_ic, test_ic
end

"""
    generate_periodic_train_test(L1,L2,t_span,number_sensors,number_test_functions,number_train_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt=1e-3,nu_val=0.1,domain="periodic")
Generate the training and testing data for a specified `pde_function_handle` for periodic boundary conditions using a Fourier spectral method. Defaults to IC \$f(\\sin^2(x/2))\$ and \$x ∈ [0,1]\$.
"""
function generate_periodic_train_test(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-3,nu_val=0.1,domain="periodic")

    dL = abs(L2-L1);
    fnc = (x)->sin(pi*x/dL)^2;

    if domain == "periodic"
        x_full = range(L1,stop = L2,length = number_sensors+1); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale)
        dL = abs(L2-L1);
        # Set up x domain and wave vector for spectral solution
        x = trapezoid(number_sensors,L1,L2)[1];
        k = reduce(vcat,(2*π/dL)*[0:number_sensors/2-1 -number_sensors/2:-1]);

    elseif domain == "full"
        x_full = range(L1,stop = L2,length = number_sensors); # Full domain for periodic number of sensors
        random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale)
        dL = abs(L2-L1);
        # Set up x domain and wave vector for spectral solution
        x = trapezoid((number_sensors-1),L1,L2)[1];
        k = reduce(vcat,(2*pi/dL)*[0:Int(number_sensors-1)/2-1 -Int(number_sensors-1)/2:-1]);
    end

    # Generate the dataset using spectral method
    t_length = t_span[2]/dt_size + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    train_ic = zeros(number_sensors,number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(number_sensors,number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    if domain == "periodic"
        @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
            interp_train_sample = random_ics[1:end-1,i];
            u_train = generate_fourier_solution(L1,L2,t_span,number_sensors,interp_train_sample,pde_function_handle,dt=dt_size)[1];
            train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train,interp_train_sample,number_solution_points);
        end
        @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
            interp_test_sample = random_ics[1:end-1,Int(number_train_functions+i)];
            u_test = generate_fourier_solution(L1,L2,t_span,number_sensors,interp_test_sample,pde_function_handle,dt=dt_size)[1];
            test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test,interp_test_sample,number_solution_points);
        end
    elseif domain == "full"
        @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
            interp_train_sample_fourier = random_ics[1:end-1,i];
            u_train_fourier = generate_fourier_solution(L1,L2,t_span,Int(number_sensors-1),interp_train_sample_fourier,pde_function_handle,dt=dt_size)[1];
            interp_train_sample = random_ics[:,i];
            u_train = periodic_fill_solution(u_train_fourier);
            train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x_full,t,u_train,interp_train_sample,number_solution_points);
        end
        @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
            interp_test_sample_fourier = random_ics[1:end-1,Int(number_train_functions+i)];
            u_test_fourier = generate_fourier_solution(L1,L2,t_span,Int(number_sensors-1),interp_test_sample_fourier,pde_function_handle,dt=dt_size)[1];
            interp_test_sample = random_ics[:,Int(number_train_functions+i)];
            u_test = periodic_fill_solution(u_test_fourier);
            test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x_full,t,u_test,interp_test_sample,number_solution_points);
        end
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(number_sensors,Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(number_sensors,Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);

    return train_data, test_data
end

"""
    generate_periodic_train_test_Adv2D(L_x,L_y,M_x,M_y,t_span,alp,number_sensors_x,number_sensors_y,number_test_functions,number_train_functions,number_solution_points,batch;dt=1e-3)

Generate the training and testing data for 2D advection with periodic boundary conditions

"""

function generate_periodic_train_test_Diff2D(L_x,L_y,M_x,M_y,t_span,nu,number_sensors_x,number_sensors_y,number_train_functions,number_test_functions,number_solution_points,batch;dt_size=1e-3)

    number_sensors = Int(number_sensors_x*number_sensors_y);

    # Generate the dataset using exact solution to the advection problem
    t_length = Int(t_span[2]/dt_size + 1);
    t_values = range(t_span[1],stop = t_span[2], length = t_length);
    
    train_ic = zeros(number_sensors,number_train_functions);
    train_loc = zeros(3,number_solution_points,number_train_functions);
    train_target = zeros(number_solution_points,number_train_functions);
    
    test_ic = zeros(number_sensors,number_test_functions);
    test_loc = zeros(3,number_solution_points,number_test_functions);
    test_target = zeros(number_solution_points,number_test_functions);

    
    dL_x = abs(L_x[2]-L_x[1]);
    if abs(dL_x) < 1.0e-12
	dL_x = 1.0;
    end

    dL_y = abs(L_y[2]-L_y[1]);
    if abs(dL_y) < 1.0e-12
	dL_y = 1.0;
    end

    x_eq = range(L_x[1],stop = L_x[2],length = number_sensors_x + 1); x_eq = x_eq[1:end-1];
    y_eq = range(L_y[1],stop = L_y[2],length = number_sensors_y + 1); y_eq = y_eq[1:end-1];

    xy_eq = zeros(number_sensors,2);
    for j = 1:number_sensors_x 
	xy_eq[(j-1)*number_sensors_y+1:j*number_sensors_y,:] = hcat(repeat([x_eq[j]],number_sensors_y,1),y_eq);
    end

    t_x_grid = zeros(number_sensors*t_length,3);
    for i = 1:t_length
	t_x_grid[(i-1)*number_sensors+1:i*number_sensors,:] = hcat(repeat([t_values[i]],number_sensors,1), xy_eq);
    end

	
    @showprogress 1 "Building training and testing datasets..." for l = 1:(number_train_functions + number_test_functions)
    	
	C_x = 2*rand(2*M_x+1,1) .- 1;
	C_y = 2*rand(2*M_y+1,1) .- 1;

	u0x(x) = C_x[1] .+ sum(cos.(2*pi*(x-L_x[1]).*(1:M_x)'/dL_x).*C_x[2:M_x+1]') .+ sum(sin.(2*pi*(x-L_x[1]).*(1:M_x)'/dL_x).*C_x[M_x+2:2*M_x+1]');
	u0y(y) = C_y[1] .+ sum(cos.(2*pi*(y-L_y[1]).*(1:M_y)'/dL_y).*C_y[2:M_y+1]') .+ sum(sin.(2*pi*(y-L_y[1]).*(1:M_y)'/dL_y).*C_y[M_y+2:2*M_y+1]');

	ux(txy) =  C_x[1] .+ sum(cos.(2*pi*(txy[2]-L_x[1]).*(1:M_x)'/dL_x).*C_x[2:M_x+1]'.*exp.(-nu[1]*txy[1]*(2*pi*(1:M_x)'/dL_x).^2)) .+ sum(sin.(2*pi*(txy[2]-L_x[1]).*(1:M_x)'/dL_x).*C_x[M_x+2:2*M_x+1]'.*exp.(-nu[1]*txy[1]*(2*pi*(1:M_x)'/dL_x).^2));
	uy(txy) =  C_y[1] .+ sum(cos.(2*pi*(txy[3]-L_y[1]).*(1:M_y)'/dL_y).*C_y[2:M_y+1]'.*exp.(-nu[2]*txy[1]*(2*pi*(1:M_y)'/dL_y).^2)) .+ sum(sin.(2*pi*(txy[3]-L_y[1]).*(1:M_y)'/dL_y).*C_y[M_y+2:2*M_y+1]'.*exp.(-nu[2]*txy[1]*(2*pi*(1:M_y)'/dL_y).^2));

	u0(x,y) = u0x.(x).*u0y.(y);
	u(txy) = ux(txy)*uy(txy);
	
	u0vals = u0(xy_eq[:,1],xy_eq[:,2]);

	
	shuffled_indices = randperm(number_sensors*t_length);
	indices = shuffled_indices[1:number_solution_points];
	# print("halooo\n")

	sol_location = t_x_grid[indices,:];
        sol_vals = zeros(number_solution_points,1);
	for jj = 1:number_solution_points
		sol_vals[jj] = u(sol_location[jj,:]);
	end  

	if l <= number_train_functions
		
		train_ic[:,l] = u0vals; 
   		train_loc[:,:,l] = sol_location'; 
   		train_target[:,l] = sol_vals;

	else
		test_ic[:,l-number_train_functions] = u0vals; 
   		test_loc[:,:,l-number_train_functions] = sol_location'; 
   		test_target[:,l-number_train_functions] = sol_vals;
	end
    end

 
    # Combine data sets from each function
    opnn_train_ic = train_ic;
    opnn_train_loc = train_loc; #reshape(hcat(train_loc...),(3,Int(number_solution_points*number_train_functions)));
    opnn_train_target = train_target; #reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    
    opnn_test_ic = test_ic;
    opnn_test_loc = test_loc; #reshape(hcat(test_loc...),(3,Int(number_solution_points*number_test_functions)));
    opnn_test_target = test_target; #reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);

    return train_data, test_data


end


"""
    generate_periodic_train_test_Adv2D(L_x,L_y,M_x,M_y,t_span,alp,number_sensors_x,number_sensors_y,number_test_functions,number_train_functions,number_solution_points,batch;dt=1e-3)

Generate the training and testing data for 2D advection with periodic boundary conditions

"""

function generate_periodic_train_test_Adv2D(L_x,L_y,M_x,M_y,t_span,alp,number_sensors_x,number_sensors_y,number_train_functions,number_test_functions,number_solution_points,batch;dt_size=1e-3)

    number_sensors = Int(number_sensors_x*number_sensors_y);

    # Generate the dataset using exact solution to the advection problem
    t_length = Int(t_span[2]/dt_size + 1);
    t_values = range(t_span[1],stop = t_span[2], length = t_length);
    
    train_ic = zeros(number_sensors,number_train_functions);
    train_loc = zeros(3,number_solution_points,number_train_functions);
    train_target = zeros(number_solution_points,number_train_functions);
    
    test_ic = zeros(number_sensors,number_test_functions);
    test_loc = zeros(3,number_solution_points,number_test_functions);
    test_target = zeros(number_solution_points,number_test_functions);

    
    dL_x = abs(L_x[2]-L_x[1]);
    if abs(dL_x) < 1.0e-12
	dL_x = 1.0;
    end

    dL_y = abs(L_y[2]-L_y[1]);
    if abs(dL_y) < 1.0e-12
	dL_y = 1.0;
    end

    x_eq = range(L_x[1],stop = L_x[2],length = number_sensors_x + 1); x_eq = x_eq[1:end-1];
    y_eq = range(L_y[1],stop = L_y[2],length = number_sensors_y + 1); y_eq = y_eq[1:end-1];

    xy_eq = zeros(number_sensors,2);
    for j = 1:number_sensors_x 
	xy_eq[(j-1)*number_sensors_y+1:j*number_sensors_y,:] = hcat(repeat([x_eq[j]],number_sensors_y,1),y_eq);
    end

    t_x_grid = zeros(number_sensors*t_length,3);
    for i = 1:t_length
	t_x_grid[(i-1)*number_sensors+1:i*number_sensors,:] = hcat(repeat([t_values[i]],number_sensors,1), xy_eq);
    end

	
    @showprogress 1 "Building training and testing datasets..." for l = 1:(number_train_functions + number_test_functions)
    	
	C_x = 2*rand(2*M_x+1,1) .- 1;
	C_y = 2*rand(2*M_y+1,1) .- 1;

	u0x(x) = C_x[1] .+ sum(cos.(2*pi*(x-L_x[1]).*(1:M_x)'/dL_x).*C_x[2:M_x+1]') .+ sum(sin.(2*pi*(x-L_x[1]).*(1:M_x)'/dL_x).*C_x[M_x+2:2*M_x+1]');
	u0y(y) = C_y[1] .+ sum(cos.(2*pi*(y-L_y[1]).*(1:M_y)'/dL_y).*C_y[2:M_y+1]') .+ sum(sin.(2*pi*(y-L_y[1]).*(1:M_y)'/dL_y).*C_y[M_y+2:2*M_y+1]');

	u0(x,y) = u0x.(x).*u0y.(y);
	u(txy) = u0x.(mod.(txy[2]-alp[1]*txy[1]-L_x[1],dL_x) .+ L_x[1]).*u0y.(mod.(txy[3]-alp[2]*txy[1]-L_y[1],dL_y) .+ L_y[1]);
	
	u0vals = u0(xy_eq[:,1],xy_eq[:,2]);

	
	shuffled_indices = randperm(number_sensors*t_length);
	indices = shuffled_indices[1:number_solution_points];
	# print("halooo\n")

	sol_location = t_x_grid[indices,:];
        sol_vals = zeros(number_solution_points,1);
	for jj = 1:number_solution_points
		sol_vals[jj] = u(sol_location[jj,:]);
	end  

	if l <= number_train_functions
		
		train_ic[:,l] = u0vals; 
   		train_loc[:,:,l] = sol_location'; 
   		train_target[:,l] = sol_vals;

	else
		test_ic[:,l-number_train_functions] = u0vals; 
   		test_loc[:,:,l-number_train_functions] = sol_location'; 
   		test_target[:,l-number_train_functions] = sol_vals;
	end
    end

 
    # Combine data sets from each function
    opnn_train_ic = train_ic;
    opnn_train_loc = train_loc; #reshape(hcat(train_loc...),(3,Int(number_solution_points*number_train_functions)));
    opnn_train_target = train_target; #reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    
    opnn_test_ic = test_ic;
    opnn_test_loc = test_loc; #reshape(hcat(test_loc...),(3,Int(number_solution_points*number_test_functions)));
    opnn_test_target = test_target; #reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);

    return train_data, test_data


end

"""
    train_model_2D(branch,trunk,n_epoch,train_data,test_data,pde_function;learning_rate=1e-5,save_at=2500,starting_epoch=0)
Train the operator neural network using the mean squared error (MSE) and Adam optimization for `n_epochs` epochs.
"""
function train_model_2D(branch,trunk,n_epoch,train_data,test_data,pde_function,learning_rate;save_at=2500,starting_epoch=0)
    
    bsz = train_data.batchsize;
    num_sen = size(train_data.data[1])[1];
    num_sol = size(train_data.data[2])[2];		    
    
    Msk = kronecker(Matrix(1.0I,bsz,bsz),ones(num_sol,1)); # Mask
    Coll = kronecker(ones(1,bsz),Matrix(1.0I,num_sol,num_sol)); # Collect, collapse

    loss(x,y,z,branch,trunk) = Flux.mse(Coll*(Msk.*(trunk(y)'*branch(x))),z)
    par = Flux.params(branch,trunk);
    opt = ADAM(learning_rate);
    @showprogress 1 "Training the model..." for i in 1:n_epoch

    for (x,y,z) in train_data

        Flux.train!((x,y,z) -> loss(x,y,z,branch,trunk),par,[(reshape(x,num_sen,bsz),reshape(y,3,num_sol*bsz),reshape(z,num_sol,bsz))],opt);
    end

    if i%save_at == 0
            save_model(branch,trunk,Int(starting_epoch+i),(pde_function*"_temp"))
            train_MSE = loss_all_2D(branch,trunk,train_data);
            test_MSE = loss_all_2D(branch,trunk,test_data);
            println("Train MSE $train_MSE")
            println("Test MSE $test_MSE")
    end
    
    end
    return branch,trunk
end


"""
    loss_all_2D(branch,trunk,data_ex)
Compute the mean squared error (MSE) for a complete dataset.
"""
function loss_all_2D(branch,trunk,data_ex)

    num_sen,num_ic = size(data_ex.data[1])[1:2];
    num_sol = size(data_ex.data[2])[2];	
    bsz = data_ex.batchsize;	

    Msk = kronecker(Matrix(1.0I,bsz,bsz),ones(num_sol,1));
    Coll = kronecker(ones(1,bsz),Matrix(1.0I,num_sol,num_sol)); 


    Errs = zeros(1,Int(num_ic/bsz));
    num_samples = Int(num_sol*num_ic);
    
    let ii = 0   
   	for (x,y,z) in data_ex
		ii += 1;
		Errs[ii] = sum((Coll*(Msk.*(trunk(reshape(y,3,num_sol*bsz))'*branch(reshape(x,num_sen,bsz)))) - reshape(z,num_sol,bsz)).^2);
	end
    end
    
    return (1/num_samples)*sum(Errs);
end



"""
    generate_periodic_train_test_muscl(L1,L2,t_span,number_sensors,number_test_functions,number_train_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,upwind_solution_points=4096,fnc=(x)->sin(x/2)^2)
Generate the training and testing data for a specified `pde_function_handle` for periodic boundary conditions using a MUSCL method. Defaults to IC \$f(\\sin^2(x/2))\$ and \$x ∈ [0,1]\$.
"""
function generate_periodic_train_test_muscl(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,upwind_solution_points=4096,fnc=(x)->sin(x/2)^2)

    x_full = range(L1,stop = L2,length = upwind_solution_points+1); # Full domain
    random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale)
    dL = abs(L2-L1);

    # Set up x domains
    x = trapezoid(number_sensors,L1,L2)[1];
    x_full = trapezoid(upwind_solution_points,L1,L2)[1];

    # Generate the dataset
    t_length = t_span[2]/dt_size + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    train_ic = zeros(Int(number_sensors),number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(Int(number_sensors),number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    @showprogress 1 "Building training dataset..." for i in 1:number_train_functions

        interp_train_sample_full = random_ics[1:end-1,i];
        u_train_full = generate_muscl_minmod_solution(L1,L2,t_span[end],upwind_solution_points,interp_train_sample_full,muscl_minmod_RHS!;dt=dt_size)
        interp_train_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_train_sample_full)));
        u_train = solution_spatial_sampling(x,x_full,u_train_full);
        train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train,interp_train_sample,number_solution_points);
    end

    @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions

        interp_test_sample_full = random_ics[1:end-1,Int(number_train_functions+i)];
        u_test_full = generate_muscl_minmod_solution(L1,L2,t_span[end],upwind_solution_points,interp_test_sample_full,muscl_minmod_RHS!;dt=dt_size);
        interp_test_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_test_sample_full)));
        u_test = solution_spatial_sampling(x,x_full,u_test_full);
        test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test,interp_test_sample,number_solution_points);
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(Int(number_sensors),Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(Int(number_sensors),Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);#,shuffle = true);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);
    return train_data, test_data
end

"""
    generate_periodic_train_test_esdirk(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,nu_val=0.1,domain="periodic",fnc=(x)->sin(x/2)^2,mode_multiplier=4)
Generate the training and testing data for a specified `pde_function_handle` for periodic boundary conditions using a Fourier spectral method and a ESDIRK ODE solver. Defaults to IC \$f(\\sin^2(x/2))\$ and \$x ∈ [0,1]\$.
"""
function generate_periodic_train_test_esdirk(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,nu_val=0.1,domain="periodic",fnc=(x)->sin(x/2)^2,mode_multiplier=4)

    x_full = range(L1,stop = L2,length = Int(mode_multiplier*number_sensors+1)); # Full domain for periodic number of sensors
    random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale)
    dL = abs(L2-L1);
    # Set up x domain and wave vector for spectral solution
    x = trapezoid(number_sensors,L1,L2)[1];
    x_full = trapezoid(Int(mode_multiplier*number_sensors),L1,L2)[1];

    # Generate the dataset using spectral method
    t_length = t_span[2]/dt_size + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    train_ic = zeros(number_sensors,number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(number_sensors,number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
        interp_train_sample_full = random_ics[1:end-1,i];
        u_train_full = generate_fourier_solution_esdirk(L1,L2,t_span,Int(mode_multiplier*number_sensors),interp_train_sample_full,pde_function_handle;dt= dt_size,nu=nu_val)[1];
        interp_train_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_train_sample_full)));
        u_train = solution_spatial_sampling(x,x_full,u_train_full);
        train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train,interp_train_sample,number_solution_points);
    end
    @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
        interp_test_sample_full = random_ics[1:end-1,Int(number_train_functions+i)];
        u_test_full = generate_fourier_solution_esdirk(L1,L2,t_span,Int(mode_multiplier*number_sensors),interp_test_sample_full,pde_function_handle;dt = dt_size,nu=nu_val)[1];
        interp_test_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_test_sample_full)));
        u_test = solution_spatial_sampling(x,x_full,u_test_full);
        test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test,interp_test_sample,number_solution_points);
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(number_sensors,Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(number_sensors,Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);

    return train_data, test_data
end

"""
    generate_periodic_train_test_implicit(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,nu_val=0.1,domain="periodic",fnc=(x)->sin(x/2)^2,mode_multiplier=4)
Generate the training and testing data for a specified `pde_function_handle` for periodic boundary conditions using a Fourier spectral method and a Crank-Nicolson solver. Defaults to IC \$f(\\sin^2(x/2))\$ and \$x ∈ [0,1]\$.
"""
function generate_periodic_train_test_implicit(L1,L2,t_span,number_sensors,number_train_functions,number_test_functions,number_solution_points,pde_function_handle;length_scale=0.5,batch=number_solution_points,dt_size=1e-4,nu_val=0.1,domain="periodic",fnc=(x)->sin(x/2)^2,mode_multiplier=4)

    x_full = range(L1,stop = L2,length = Int(mode_multiplier*number_sensors+1)); # Full domain for periodic number of sensors
    random_ics = generate_periodic_functions(fnc,x_full,Int(number_train_functions + number_test_functions),length_scale)
    dL = abs(L2-L1);
    # Set up x domain and wave vector for spectral solution
    x = trapezoid(number_sensors,L1,L2)[1];
    x_full = trapezoid(Int(mode_multiplier*number_sensors),L1,L2)[1];

    # Generate the dataset using spectral method
    t_length = t_span[2]/dt_size + 1;
    t = range(t_span[1],stop = t_span[2], length = Int(t_length));
    train_ic = zeros(number_sensors,number_solution_points,number_train_functions);
    train_loc = zeros(2,number_solution_points,number_train_functions);
    train_target = zeros(1,number_solution_points,number_train_functions);
    test_ic = zeros(number_sensors,number_solution_points,number_test_functions);
    test_loc = zeros(2,number_solution_points,number_test_functions);
    test_target = zeros(1,number_solution_points,number_test_functions);

    @showprogress 1 "Building training dataset..." for i in 1:number_train_functions
        interp_train_sample_full = random_ics[1:end-1,i];
        u_train_full = generate_fourier_solution_implicit(L1,L2,t_span,Int(mode_multiplier*number_sensors),interp_train_sample_full,pde_function_handle;dt= dt_size,nu=nu_val)[1];
        interp_train_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_train_sample_full)));
        u_train = solution_spatial_sampling(x,x_full,u_train_full);
        train_ic[:,:,i], train_loc[:,:,i], train_target[:,:,i] = solution_extraction(x,t,u_train,interp_train_sample,number_solution_points);
    end
    @showprogress 1 "Building testing dataset..." for i in 1:number_test_functions
        interp_test_sample_full = random_ics[1:end-1,Int(number_train_functions+i)];
        u_test_full = generate_fourier_solution_implicit(L1,L2,t_span,Int(mode_multiplier*number_sensors),interp_test_sample_full,pde_function_handle;dt = dt_size,nu=nu_val)[1];
        interp_test_sample = transpose(solution_spatial_sampling(x,x_full,transpose(interp_test_sample_full)));
        u_test = solution_spatial_sampling(x,x_full,u_test_full);
        test_ic[:,:,i], test_loc[:,:,i], test_target[:,:,i] = solution_extraction(x,t,u_test,interp_test_sample,number_solution_points);
    end

    # Combine data sets from each function
    opnn_train_ic = reshape(hcat(train_ic...),(number_sensors,Int(number_solution_points*number_train_functions)));
    opnn_train_loc = reshape(hcat(train_loc...),(2,Int(number_solution_points*number_train_functions)));
    opnn_train_target = reshape(hcat(train_target...),(1,Int(number_solution_points*number_train_functions)));
    opnn_test_ic = reshape(hcat(test_ic...),(number_sensors,Int(number_solution_points*number_test_functions)));
    opnn_test_loc = reshape(hcat(test_loc...),(2,Int(number_solution_points*number_test_functions)));
    opnn_test_target = reshape(hcat(test_target...),(1,Int(number_solution_points*number_test_functions)));

    train_data = DataLoader(opnn_train_ic, opnn_train_loc, opnn_train_target, batchsize=batch);
    test_data = DataLoader(opnn_test_ic, opnn_test_loc, opnn_test_target, batchsize=batch);

    return train_data, test_data
end

"""
    save_model(branch,trunk,n_epoch,loss_all_train,loss_all_test,pde_function)
Save the trained `branch` and `trunk` neural networks and the training and testing loss history.
"""
function save_model(branch,trunk,n_epoch,pde_function)
    @save @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @save @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
end

"""
    load_model(n_epoch,pde_function)
Load the trained `branch` and `trunk` neural networks.
"""
function load_model(n_epoch,pde_function)
    @load @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @load @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    return branch, trunk
end

"""
    load_branch(n_epoch,pde_function)
Load the trained `branch` neural networks.
"""
function load_branch(n_epoch,pde_function)
    @load @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    return branch
end

"""
    load_trunk(n_epoch,pde_function)
Load the trained `trunk` neural networks.
"""
function load_trunk(n_epoch,pde_function)
    @load @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    return trunk
end

"""
    save_data(train_data,test_data,number_train_functions,number_test_functions,number_solution_points,pde_function)
Save the `train_data` and `test_data`.
"""
function save_data(train_data,test_data,number_train_functions,number_test_functions,number_solution_points,pde_function)
    train_ic = train_data.data[1];
    train_loc = train_data.data[2];
    train_sol = train_data.data[3];
    test_ic = test_data.data[1];
    test_loc = test_data.data[2];
    test_sol = test_data.data[3];
    @save @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @save @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @save @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @save @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @save @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @save @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
end

"""
    load_data(n_epoch,number_train_functions,number_test_functions,pde_function)
Load the trained `branch` and `trunk` neural networks along with the `train_data` and `test_data`.
"""
function load_data(n_epoch,number_train_functions,number_test_functions,pde_function)
    @load @sprintf("branch_epochs_%i_%s.bson",n_epoch,pde_function) branch
    @load @sprintf("trunk_epochs_%i_%s.bson",n_epoch,pde_function) trunk
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @load @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @load @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @load @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
    return branch, trunk, train_ic, train_loc, train_sol, test_ic, test_loc, test_sol#
end

"""
    load_data_train_test(number_train_functions,number_test_functions,pde_function)
Load the `train_data` and `test_data`.
"""
function load_data_train_test(number_train_functions,number_test_functions,pde_function)
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("train_loc_data_%i_%s.bson",number_train_functions,pde_function) train_loc
    @load @sprintf("train_target_data_%i_%s.bson",number_train_functions,pde_function) train_sol
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    @load @sprintf("test_loc_data_%i_%s.bson",number_test_functions,pde_function) test_loc
    @load @sprintf("test_target_data_%i_%s.bson",number_test_functions,pde_function) test_sol
    return train_ic, train_loc, train_sol, test_ic, test_loc, test_sol
end

"""
    load_data_initial_conditions(number_train_functions,number_test_functions,pde_function)
Load the initial conditions from the `train_data` and `test_data`.
"""
function load_data_initial_conditions(number_train_functions,number_test_functions,pde_function)
    @load @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @load @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
    return train_ic, test_ic
end

"""
    save_data_initial_conditions(number_train_functions,number_test_functions)
Saves the initial conditions from the `train_ic` and `test_ic`.
"""
function save_data_initial_conditions(train_ic,test_ic,number_train_functions,number_test_functions,pde_function)
    @save @sprintf("train_ic_data_%i_%s.bson",number_train_functions,pde_function) train_ic
    @save @sprintf("test_ic_data_%i_%s.bson",number_test_functions,pde_function) test_ic
end
