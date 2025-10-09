% This script solves the inverse kernel problem for the Fredholm Integral Equation. A shallow %neural network is used to model the kernel and training is performed by passing through the %Fredholm Neural Network. Change the instances to only run a single (or more) instances.


% Total number of training instances
total_instances = 50; 
progress_file = 'training_progress_test_another_reg.mat';

% Check if a progress file exists for resuming.
if exist(progress_file, 'file')
    loaded = load(progress_file, 'MSEs', 'models', 'lastInstance');
    MSEs = loaded.MSEs;
    models = loaded.models;
    start_instance = loaded.lastInstance + 1;
    fprintf('Resuming training from instance %d...\n', start_instance);
else
    MSEs = zeros(total_instances, 1);
    models = cell(total_instances, 1);
    start_instance = 1;
end

% Ensure that required variables from the forward Fredholm NN are present, i.e., the constructed  % data. Required variables are (final_prediction, y, N, dy) as produced by the %Forward_Fredholm_Integral_Equation.m script. These have to be in the workspace.

%% Training Loop
for inst = start_instance:total_instances
    fprintf('\n--- Training Instance %d ---\n', inst);

 
    x_train = y;                          % Training grid (row vector)
    % f_target = final_prediction(:)';        % Target f(x) as a row vector
    f_target = final_prediction; 
    N_data = N;                           % Number of training points
    dx = dy;                              % Use dy from run_fred as the integration step
    % For integration, we assume f(z) = f_target (evaluated on x_train)
    fz = f_target;
    fprintf('Size of f_target: %s\n', mat2str(size(f_target)));
    
    %% 2. Define the Additive Function
    % This function appears outside the integral in the FNN.
    % additive_part = @(x) sin(25*x) + sin(7*x);
    additive_part = @(value) sin(25 * value) + sin(7 * value);
    
    %% 3. Instantiate the Kernel Network
    % This network will learn the kernel function.
    N_neurons = 20;
    net_kernel = feedforwardnet(N_neurons, 'trainlm');
    dummy_input = rand(2, 10);
    dummy_output = rand(1, 10);
    net_kernel = configure(net_kernel, dummy_input, dummy_output);
    
    net_kernel.layers{1}.transferFcn = 'tansig';
    net_kernel.layers{2}.transferFcn = 'purelin';
    % 
    % 
    % % After configuring the network:
    
    
    %% 4. Instantiate the Fredholm Neural Network Model
    % Create a grid_dictionary from x_train. We assume the same grid for each layer.
    K_train = 15;  % Number of layers (excluding layer_0)
    input_size = numel(x_train);
    output_size = numel(x_train);
    
    grid_dictionary = struct();
    for i = 0:K_train
        field_name = sprintf('layer_%d', i);
        grid_dictionary.(field_name) = x_train;
    end
    
    % Supply a dummy kernel function, which will be replaced later, during training.
    dummy_kernel = @(in_val, out_val) 0;
    fred_model = FredholmNeuralNetwork(grid_dictionary, dummy_kernel, additive_part, additive_part, dx, K_train, input_size, output_size);
    
    %% 5. Define the Error Function for lsqnonlin with L_2 Regularization
    % This error function:
    %   1. Updates the kernel network with weights wb.
    %   2. Defines a new kernel function using evaluate_kernel.
    %   3. Updates fred_model.kernel with this new kernel.
    %   4. Runs fred_model.forward(x_train) to obtain f_FNN(x).
    %   5. Returns the error vector: f_FNN(x) - f_target(x), augmented with a
    %      regularization term on wb.
    model_error = @(wb) error_helper(net_kernel, wb, fred_model, x_train, f_target, dx, additive_part);
    
    %% 6. Optimize the Kernel Network Using lsqnonlin
    wb0 = getwb(net_kernel);
    options = optimoptions('lsqnonlin', ...
        'Display', 'iter-detailed', ...
        'Algorithm', 'levenberg-marquardt', ...
        'FunctionTolerance', 1e-5, ...
        'MaxIterations', 300);
    [wb_opt, resnorm, residual, exitflag, output] = lsqnonlin(model_error, wb0, [], [], options);
    net_kernel = setwb(net_kernel, wb_opt);
    
    %% 7. Evaluate the Final Model Performance
    % Use the trained kernel network to define the new kernel function,
    % update fred_model.kernel, and run the FNN forward pass.
    f_hat = compute_model_output(net_kernel, fred_model, x_train, dx, additive_part);
    % f_hat = f_hat(:)';
    mse_final = immse(f_hat, f_target);
    fprintf('Instance %d: Final MSE = %e\n', inst, mse_final);
    
    % save('trainedModel.mat', 'net_kernel', 'mse_final');

    % Save the final MSE and the trained kernel network for this instance.
    MSEs(inst) = mse_final;
    models{inst} = net_kernel;


        %% 8. Save Progress Every 2 Instances or on the Final Instance
    if mod(inst, 2) == 0 || inst == total_instances
        lastInstance = inst;
        save(progress_file, 'MSEs', 'models', 'lastInstance');
        fprintf('Progress saved at instance %d.\n', inst);
    end
end

fprintf('Training complete. Results saved in %s.\n', progress_file);
    
    
    % %% 8. Plot the Results
    % figure;
    % plot(x_train, f_target, 'b', 'LineWidth', 2); hold on;
    % plot(x_train, f_hat, 'g--', 'LineWidth', 2);
    % legend('Target f(x)', 'FNN Output with Learned Kernel');
    % xlabel('x'); ylabel('f(x)');
    % title('FNN Output using Learned Kernel vs. Target');
    % grid on;
    % 
    % figure;
    % plot(x_train, abs(f_hat - f_target), 'r--', 'LineWidth', 2);
    % xlabel('x'); ylabel('|f_{hat}(x)-f(x)|');
    % title('Absolute Error');
    % grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% --- Local Functions ---

function err = error_helper(net_kernel, wb, fred_model, x_train, f_target, dx, additive_part)
%ERROR_HELPER Computes the error vector for lsqnonlin optimization with L₂ regularization.
%
% For each x in x_train, it does the following:
%  1. Updates the kernel network with weights wb.
%  2. Defines a new kernel function using evaluate_kernel.
%  3. Sets fred_model.kernel to this new kernel.
%  4. Runs fred_model.forward(x_train) to obtain f_FNN(x).
%  5. Returns the error vector: f_FNN(x) - f_target(x) augmented with
%     a regularization term on wb.
%
% The overall error vector is:
%   [ f_FNN(x) - f_target(x) ; sqrt(reg_lambda) * wb ]
    
    % Update the kernel network with current weights.
    net_kernel = setwb(net_kernel, wb);
    
    % Define the new kernel function using the evaluate_kernel helper.
    new_kernel = @(in_val, out_val) evaluate_kernel(net_kernel, in_val, out_val);
    
    % Update fred_model's kernel.
    fred_model.kernel = new_kernel;
    
    % Run the forward pass of the Fredholm NN.
    [f_FNN, ~] = fred_model.forward(x_train);
    % f_FNN = f_FNN(:)';
    % fprintf('Size of f_FNN: %s\n', mat2str(size(f_FNN)));

    % Compute the data error.
    err_data = f_FNN - f_target;

    % L₂ regularization on the kernel network weights.
    reg_lambda = 1e-6;  % Adjust as needed.
    % err_reg = sqrt(reg_lambda) * wb(:);
    %K_vals = evaluate_kernel(net_kernel, x_train, x_train);  % produces an N×N matrix
    %err_reg = sqrt(reg_lambda) * K_vals(:);

    Kmat = evaluate_kernel(net_kernel, x_train, x_train);        % N×N
    integral_part = (Kmat * f_target) * dx;                            % N×1
    err_reg  = sqrt(reg_lambda) * (f_target(:) - integral_part - additive_part(x_train(:)));
    
    % Concatenate the data error and regularization term.
    err = [err_data(:); err_reg];
end

function f_hat = compute_model_output(net_kernel, fred_model, x_train, dx, additive_part)
%COMPUTE_MODEL_OUTPUT Computes the FNN output using the current kernel network.
%
%  1. Defines a new kernel function using net_kernel and evaluate_kernel.
%  2. Updates fred_model.kernel with the new kernel function.
%  3. Runs fred_model.forward(x_train) to compute f_hat(x).
%
% Returns f_hat, the output of the Fredholm NN.
    
    % Define the new kernel function.
    new_kernel = @(in_val, out_val) evaluate_kernel(net_kernel, in_val, out_val);
    
    % Update fred_model's kernel.
    fred_model.kernel = new_kernel;
    
    % Run the forward pass.
    [f_hat, ~] = fred_model.forward(x_train);
    
    
end

function out = evaluate_kernel(net_kernel, in_val, out_val)
%EVALUATE_KERNEL Evaluates the kernel network for given in_val and out_val.
%
% in_val and out_val are vectors. This function uses ndgrid to create all
% combinations of in_val and out_val, passes them to net_kernel, and reshapes
% the result into a matrix of size length(in_val) x length(out_val).
%
% The output is the learned kernel matrix K(in_val, out_val).

    % Ensure inputs are column vectors.
    % in_val = in_val(:);
    % out_val = out_val(:);
    in_val = in_val(:).';   % Force row vector (1 x n)
    out_val = out_val(:);    % Force column vector (m x 1)
    
    % Generate grid of (x,z) pairs.
    % [X, Z] = ndgrid(in_val, out_val);  % X and Z are (n x m) matrices.
    [X, Z] = ndgrid(out_val, in_val); % X and Z are m x n

    
    % Create input for net_kernel: a 2 x (n*m) matrix.
    NN_input = [X(:)'; Z(:)'];
    
    % Evaluate the kernel network.
    out_vector = net_kernel(NN_input);
    
    % Reshape into an n x m matrix.
    out = reshape(out_vector, size(X));
end
