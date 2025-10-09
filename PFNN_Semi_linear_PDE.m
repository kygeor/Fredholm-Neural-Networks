%% Iterative solver for semi-linear PDE, similar to the approach in the Python notebooks. 
%% The script uses the simple class (not the "dense" versions).

% Clear workspace and close figures

% Parameters
K = 100;
phi_0 = 0.0;
phi_n = 2.0 * pi;
phi_steps = 150;
dphi = (phi_n - phi_0) / phi_steps;
phi = phi_0:dphi:phi_n-dphi;

% Create phi_grid_dict
phi_grid_dict = struct();
for i = 0:K
    phi_grid_dict.(sprintf('layer_%d', i)) = phi;
end

% Define grid for prediction
phi_grid = phi_grid_dict.layer_0;  % All layers share the same phi grid
r_out = linspace(0, 1, 150);
theta_out = phi_grid;

% Initialize u_pred
u_pred = zeros(length(r_out), length(theta_out));

% Compute integral of lambda*fund outside the loop (doesn't depend on u_pred)
disp('Computing precomputed integral for sqrt(lambda)*fundamental...');
tic;
precomputed_integral_entire_domain_fund = parallelIntegrateMeshgrid(r_out, theta_out);
toc;

%% Model parameters
input_size = length(phi_grid);
output_size = length(theta_out);
km_constant = 0.3;
lambda_value = 1.0;
iterations = 12;

% Iterative solution
for it = 1:iterations
    fprintf('Iteration %d/%d\n', it, iterations);
    start_time = tic;
    
    % Compute integral of fund*source(u)
    disp('Computing integral of fundamental*source...');
    precomputed_integral_entire_domain_source = computeIntegralFundSource(...
        r_out, theta_out, dphi, @fundamentalFn, @sourceNonLinear, ...
        lambda_value, r_out, theta_out, u_pred);
    
    % Take the value corresponding to r_out = 1 for Fredholm NN
    precomputed_integrals_p = precomputed_integral_entire_domain_source(end, :);
    
    % Convert results to map for quick lookup
    precomputed_integral_map = containers.Map(num2cell(phi_grid), num2cell(precomputed_integrals_p));
    
    disp(['Integral computation time: ' num2str(toc(start_time)) ' seconds']);
    
    % Start timer for model computation
    t_start = tic;
    
    % Instantiate the Fredholm model
    fredholm_model = FredholmNeuralNetwork_PDE(...
        phi_grid_dict, @kernel, dphi, K, input_size, output_size, km_constant, ...
        @funcFn, precomputed_integral_map);
    
    % Instantiate the LimitInformed model
    limit_informed_model = PotentialFredholmNeuralNetwork_Helmholtz(...
        fredholm_model, @diffPotentialsLimit, @potentialBoundary, ...
        precomputed_integral_entire_domain_source, precomputed_integral_entire_domain_fund, ...
        lambda_value, false);
    
    % Forward pass
    output = limit_informed_model.forward(theta_out', r_out, theta_out, phi_grid, dphi);
    
    % Update u_pred for next iteration
    u_pred = output;
    
    if mod(it, 3) == 0
        % Visualization of current solution
        [R_out_grid, Theta_out_grid] = ndgrid(r_out, theta_out);
        figure('Position', [100, 100, 800, 600]);
        contourf(Theta_out_grid, R_out_grid, u_pred, 50, 'LineColor', 'none');
        colorbar_obj = colorbar;
        colorbar_obj.Label.String = '$\tilde{u}(r,\phi)$';
        colorbar_obj.Label.Interpreter = 'latex';
        colorbar_obj.Label.FontSize = 18;
        xlabel('\theta', 'FontSize', 18);
        ylabel('r', 'FontSize', 18);
        title(['Semilinear PDE Solution - Iteration ' num2str(it)], 'FontSize', 16);
        set(gca, 'FontSize', 16);
        drawnow;
    end
    % Clear variables to free memory
    clear precomputed_integral_map;
    
    % Display computation time
    comp_time = toc(t_start);
    fprintf('Computation time for iteration %d: %.2f seconds\n', it, comp_time);
end

% Final visualization
[R_out_grid, Theta_out_grid] = ndgrid(r_out, theta_out);
figure('Position', [100, 100, 800, 600]);
contourf(Theta_out_grid, R_out_grid, u_pred, 50, 'LineColor', 'none');
colorbar_obj = colorbar;
colorbar_obj.Label.String = '$\tilde{u}(r,\phi)$';
colorbar_obj.Label.Interpreter = 'latex';
colorbar_obj.Label.FontSize = 18;
xlabel('\theta', 'FontSize', 18);
ylabel('r', 'FontSize', 18);
title('Final Semilinear PDE Solution', 'FontSize', 16);
set(gca, 'FontSize', 16);

%% Contour plot for visualization of absolute error in (r, theta) space

[R_grid, Theta_grid] = ndgrid(r_out, theta_out);
figure('Position',[300,300,800,600]);
difference = abs(u_pred - laplace_solution_polar(R_grid, Theta_grid));
contourf(Theta_grid, R_grid, difference, 50, 'LineColor','none');
colorbar;
xlabel('\phi', 'FontSize', 18);
ylabel('r', 'FontSize', 18);
title('Absolute Error: $|\tilde{u}(r,\phi) - u(r,\phi)|$', 'Interpreter', 'latex', 'FontSize', 18);
set(gca, 'FontSize', 16);


%% Local functions

%% Integration functions for semilinear PDE solver

% Global parameter
lambda_value = 1.0;
tol = 1e-8;

function fundamental = fundamentalFnScalar(r_2, theta, r_out, theta_out)
    % Scalar version of the fundamental function for numerical integration
    lambda_value = 1.0;
    norm = sqrt((r_out*cos(theta_out) - r_2*cos(theta)).^2 + (r_out*sin(theta_out) - r_2*sin(theta)).^2);
    fundamental = (-1/(2*pi)) * besselk(0, sqrt(lambda_value) * norm);
    
    % Debug for NaN or Inf values (uncomment if needed)
    % if isnan(fundamental) || isinf(fundamental)
    %     fprintf('Warning: Invalid value at r_2=%f, theta=%f, r_out=%f, theta_out=%f\n', ...
    %         r_2, theta, r_out, theta_out);
    % end
end

function val = integrandScalarBoundary(r_2, theta, r_out, theta_out)
    % Integrand for the second integral:
    % sqrt(lambda_value) * fundamental_fn_scalar(...) * r_2
    lambda_value = 1.0;
    fundamental = fundamentalFnScalar(r_2, theta, r_out, theta_out);
    val = sqrt(lambda_value) * fundamental * r_2;
end

function inner = innerIntegration(r2, r_out, theta_out, tol, integrand_fn)
    % Performs the inner integration over theta for a fixed r_2
    % If r_2 is close to r_out (within tol), the singular point theta_out is specified
    integrand_theta = @(th) integrand_fn(r2, th, r_out, theta_out);
    if abs(r2 - r_out) < tol
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, 'Waypoints', theta_out, 'ArrayValued', true);
    else
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, 'ArrayValued', true);
    end
end

function result = fullIntegration(integrand_fn, r_out, theta_out, tol)
    % Performs the full integration by first integrating over theta (inner)
    % and then over r_2 (outer)
    f_r2 = @(r2) innerIntegration(r2, r_out, theta_out, tol, integrand_fn);
    result = integral(f_r2, 0, 1, 'AbsTol', tol, 'RelTol', tol, 'Waypoints', r_out, 'ArrayValued', true);
end

function result = integrateForOutValueScalar(r_out, theta_out, tol)
    % Performs the sequential integration for a single (r_out, theta_out) value
    result = fullIntegration(@integrandScalarBoundary, r_out, theta_out, tol);
end

function results = parallelIntegrateMeshgrid(r_out_vals, theta_out_vals)
    % Parallel integration for scalar integrals over a meshgrid of (r_out, theta_out)
    % values using the sequential two-step integration
    tol = 1e-8;
    [R_out, Theta_out] = ndgrid(r_out_vals, theta_out_vals);
    Rf = R_out(:); Tf = Theta_out(:);
    results_flat = zeros(numel(Rf), 1);
    
    parfor i = 1:numel(Rf)
        result = integrateForOutValueScalar(Rf(i), Tf(i), tol);
        results_flat(i) = result;
    end
    
    results = reshape(results_flat, size(R_out));
end

%% Functions for PDE 

% Function for BC
function func = funcFn(outValue)
    func = zeros(size(outValue));
end

% Function for kernel
function k = kernel(inValue, outValue)
    lambda_value = 1.0;
    tol = 1e-08;
    inValue = inValue(:);
    outValue = outValue(:)';
    
    diffX = cos(outValue) - cos(inValue);
    diffY = sin(outValue) - sin(inValue);
    norm = sqrt(diffX.^2 + diffY.^2);
    
    kernelArray = zeros(size(norm));
    
    eqMask = abs(inValue - outValue) < tol;
    kernelArray(eqMask) = 1.0 / (4.0 * pi);
    
    lamSqrt = sqrt(lambda_value);
    elseMask = ~eqMask;
    if any(elseMask(:))
        derivBessel = -besselk(1, lamSqrt * norm(elseMask));
        coeff = -lamSqrt / (4.0 * pi);
        kernelElse = coeff * norm(elseMask) .* derivBessel;
        kernelArray(elseMask) = kernelElse;
    end
    
    k = -2 * kernelArray;
end


function fundamental = fundamentalFn(out_value, R_2, THETA, r_out)
    % MATLAB version of fundamental_fn from Python
    % Computes the 2D modified Helmholtz fundamental solution with polar Jacobian
    %
    % Parameters:
    %   out_value - Angular coordinate(s) of the 'field' point, theta_out
    %   R_2 - Radial coordinate(s) of the 'source' point, r_y
    %   THETA - Angular coordinate(s) of the 'source' point, theta_y
    %   r_out - Radial coordinate(s) of the 'field' point (default=1.0)
    %
    % Returns:
    %   fundamental - The fundamental solution times r_y
    
    lambda_value = 1.0;
    
    % Handle input shapes for broadcasting
    % Reshape out_value and r_out to column vectors
    % out_value = out_value(:);
    % r_out = r_out(:);
    % 
    % % Reshape R_2 and THETA to row vectors
    % R_2 = R_2(:)';
    % THETA = THETA(:)';
    % 
    % % Create meshgrids for broadcasting
    % % [Out_Value, R2_grid] = ndgrid(out_value, R_2);
    % % [R_Out, Theta_grid] = ndgrid(r_out, THETA);
    % 
    % disp(size(Out_Value));
    % disp(size(Theta_grid));
    % 
    % % Compute the distance norm
    % diff_x = R_Out .* cos(Out_Value) - R2_grid .* cos(Theta_grid);
    % diff_y = R_Out .* sin(Out_Value) - R2_grid .* sin(Theta_grid);

    diff_x = r_out    .* cos(out_value)  -  R_2 .* cos(THETA);
    diff_y = r_out    .* sin(out_value)  -  R_2 .* sin(THETA);
    
    norm_vals = sqrt(diff_x.^2 + diff_y.^2);
    
    % Evaluate Bessel function K0
    arg = sqrt(lambda_value) * norm_vals;
    fundamental = (-1/(2*pi)) * besselk(0, arg);
    % disp(size(fundamental));
end

%% Function for iterative source term

function term = sourceNonLinear(r_space, theta_space, u_pred, lambda_value)
    % MATLAB version of source_non_linear_torch function
    % Computes the nonlinear source for all (r, theta) points
    %
    % Arguments:
    %   r_space       - 1D array of radial coordinates [Nr]
    %   theta_space   - 1D array of angular coordinates [Nθ]
    %   u_pred        - 2D array storing predicted u [Nr, Nθ]
    %   lambda_value  - scalar, the parameter λ in the PDE
    %
    % Returns:
    %   term - 2D array of source term [Nr, Nθ]
    
    % Build the meshgrid for r and theta
    [r_grid, theta_grid] = ndgrid(r_space, theta_space);
    
    % Compute the PDE source term
    % term = -λ*u - exp(u) + (exp(1 - r^2) - 4)
    term = -lambda_value * u_pred - exp(u_pred) + (exp(1.0 - r_grid.^2) - 4.0);
end

%% Function for integrals using trapezoidal for the iterative scheme

function [fund_val, src_val, R_2_expanded, r_2_vals, theta_vals] = prepareIntegralData(r_out, theta_out, grid_step, fundamental_fn, source_non_linear, lambda_value, r_space, theta_space, u_pred)
    % Internal helper function that sets up everything needed for the integral calculations
    %
    % Parameters:
    %   r_out - Radial coordinates for output points
    %   theta_out - Angular coordinates for output points
    %   grid_step - Grid step size for integration
    %   fundamental_fn - Function handle for fundamental solution
    %   source_non_linear - Function handle for nonlinear source term
    %   lambda_value - Scalar lambda value
    %   r_space - Radial coordinates for the mesh
    %   theta_space - Angular coordinates for the mesh
    %   u_pred - Predicted solution values at mesh points
    %
    % Returns:
    %   fund_val - Fundamental solution array of shape [R, T, r2_steps, theta_steps]
    %   src_val - Source term array of shape [1, 1, r2_steps, theta_steps]
    %   R_2_expanded - Radial coordinate mesh expanded to shape [1, 1, r2_steps, theta_steps]
    %   r_2_vals - 1D vector of radii for the internal integral
    %   theta_vals - 1D vector of angles for the internal integral
    
    % Get step sizes
    dr_y = r_out(2) - r_out(1);
    dtheta_y = theta_out(2) - theta_out(1);
    
    % Ensure inputs are column vectors
    r_2_vals = r_space(:);
    theta_vals = theta_space(:);
    
    % 1) Create 2D mesh for r2, theta
    [R_2, THETA] = ndgrid(r_2_vals, theta_vals);
    % shape: [r2_steps, theta_steps]
    
    % 2) Prepare broadcast expansions for the output side
    [r_grid, th_grid] = ndgrid(r_out, theta_out);
    % shape: [R, T]
    
    % Reshape for broadcasting
    R_out = size(r_out(:), 1);
    T_out = size(theta_out(:), 1);
    r2_steps = size(r_2_vals, 1);
    theta_steps = size(theta_vals, 1);
    
    % Create expanded versions for broadcasting
    R_2_expanded = reshape(R_2, [1, 1, r2_steps, theta_steps]);
    THETA_expanded = reshape(THETA, [1, 1, r2_steps, theta_steps]);
    r_out_expanded = reshape(r_grid, [R_out, T_out, 1, 1]);
    th_out_expanded = reshape(th_grid, [R_out, T_out, 1, 1]);
    
    % 3) Evaluate the fundamental solution
    fund_val = fundamental_fn(th_out_expanded, R_2_expanded, THETA_expanded, r_out_expanded);
    % shape: [R, T, r2_steps, theta_steps]
    
    % Handle NaN/Inf if necessary
    mask = isinf(fund_val) | isnan(fund_val);
    if any(mask(:))
        adjusted_fundamental = fund_val;
        inf_mask = isinf(fund_val);
        nan_mask = isnan(fund_val);
        
        % Example small offset correction
        if any(inf_mask(:)) || any(nan_mask(:))
            fund_correction = fundamental_fn(th_out_expanded, ...
                R_2_expanded + dr_y/2, ...
                THETA_expanded + dtheta_y/2, ...
                r_out_expanded);
            adjusted_fundamental(inf_mask | nan_mask) = fund_correction(inf_mask | nan_mask);
        end
        fund_val = adjusted_fundamental;
    end
    
    % 4) Evaluate the source term
    src_2d = source_non_linear(r_2_vals, theta_vals, u_pred, lambda_value);
    
    % Reshape for broadcasting
    src_val = reshape(src_2d, [1, 1, r2_steps, theta_steps]);
end

function integral_1 = computeIntegralFundSource(r_out, theta_out, grid_step, fundamental_fn, source_non_linear, lambda_value, r_space, theta_space, u_pred)
    % Computes the integral of (fundamental_fn * source_non_linear * r2) over the domain
    %
    % Parameters:
    %   r_out - Radial coordinates for output points
    %   theta_out - Angular coordinates for output points
    %   grid_step - Grid step size for integration
    %   fundamental_fn - Function handle for fundamental solution
    %   source_non_linear - Function handle for nonlinear source term
    %   lambda_value - Scalar lambda value
    %   r_space - Radial coordinates for the mesh
    %   theta_space - Angular coordinates for the mesh
    %   u_pred - Predicted solution values at mesh points
    %
    % Returns:
    %   integral_1 - Integrated values of shape [R, T]

    % 1) Use helper to get all necessary data
    [fund_val, src_val, R_2_expanded, r_2_vals, theta_vals] = prepareIntegralData(...
        r_out, theta_out, grid_step, fundamental_fn, source_non_linear, ...
        lambda_value, r_space, theta_space, u_pred);

    % 2) Build integrand => fundamental * source * r_2
    integrand_1 = fund_val .* src_val .* R_2_expanded;

    % 3) Trapezoidal integration w.r.t. theta, then r_2
    % Assume theta_vals are uniformly spaced
    delta_theta = theta_vals(2) - theta_vals(1);  % Constant step size
    integral_theta_1_simple = sum(integrand_1, 4) * delta_theta;  % Sum along 4th dimension (theta)
    delta_r2 = r_2_vals(2) - r_2_vals(1);  % Constant step size
    integral_1 = sum(integral_theta_1_simple, 3) * delta_r2;  % Sum along 3rd dimension (r2)

    % Alternative using trapz:
    % integral_theta_1 = trapz(theta_vals, integrand_1, 4);  % [R, T, r2_steps]
    % integral_1 = trapz(r_2_vals, integral_theta_1, 3);     % [R, T]
end



function integral_2 = computeIntegralSqrtlambdaFund(r_out, theta_out, grid_step, fundamental_fn, source_non_linear, lambda_value, r_space, theta_space, u_pred)
    % Computes the integral of (sqrt(lambda) * fundamental_fn * r2) over the domain
    %
    % Parameters:
    %   r_out - Radial coordinates for output points
    %   theta_out - Angular coordinates for output points
    %   grid_step - Grid step size for integration
    %   fundamental_fn - Function handle for fundamental solution
    %   source_non_linear - Function handle for nonlinear source term
    %   lambda_value - Scalar lambda value
    %   r_space - Radial coordinates for the mesh
    %   theta_space - Angular coordinates for the mesh
    %   u_pred - Predicted solution values at mesh points
    %
    % Returns:
    %   integral_2 - Integrated values of shape [R, T]
    
    % 1) Use helper to get all necessary data
    [fund_val, ~, R_2_expanded, r_2_vals, theta_vals] = prepareIntegralData(...
        r_out, theta_out, grid_step, fundamental_fn, source_non_linear, ...
        lambda_value, r_space, theta_space, u_pred);
    
    lam_sqrt = sqrt(lambda_value);
    
    % 2) Build integrand => sqrt(lambda) * fundamental * r_2
    integrand_2 = lam_sqrt * fund_val .* R_2_expanded;
    
    % 3) Trapezoidal integration w.r.t. theta, then r_2
    integral_theta_2 = trapz(theta_vals, integrand_2, 4);  % [R, T, r2_steps]
    integral_2 = trapz(r_2_vals, integral_theta_2, 3);     % [R, T]
end





%% Function for diffPotentialsLimit in LINN

function kernelLimit = diffPotentialsLimit(phiIntegral, r_out, theta_out)
    atol = 1e-08;
    r_out = r_out(:);
    theta_out = theta_out(:);
    phiIntegral = phiIntegral(:);
    
    [R, TH, PH] = ndgrid(r_out, theta_out, phiIntegral);
    lambda_value = 1.0;
    
    lamSqrt = sqrt(lambda_value);
    coeffInner = -lamSqrt / (2.0 * pi);
    
    coeffN = cos(PH) .* (cos(PH) - R .* cos(TH)) + sin(PH) .* (sin(PH) - R .* sin(TH));
    
    diffXInner = cos(PH) - R .* cos(TH);
    diffYInner = sin(PH) - R .* sin(TH);
    normInner = sqrt(diffXInner.^2 + diffYInner.^2);
    
    besselInner = -besselk(1, lamSqrt * normInner);
    
    kernelInner = coeffInner * besselInner .* (coeffN ./ normInner);
    
    eqMaskInner = abs(PH - TH) < atol;
    kernelInner(eqMaskInner) = 1.0 / (4.0 * pi);
    
    coeffBoundary = -lamSqrt / (4.0 * pi);
    
    diffXBdry = cos(PH) - cos(TH);
    diffYBdry = sin(PH) - sin(TH);
    normBdry = sqrt(diffXBdry.^2 + diffYBdry.^2);
    
    besselBdry = -besselk(1, lamSqrt * normBdry);
    
    kernelBoundary = coeffBoundary * besselBdry .* normBdry;
    
    eqMaskBdry = abs(PH - TH) < atol;
    kernelBoundary(eqMaskBdry) = 1.0 / (4.0 * pi);
    
    kernelLimit = kernelInner - kernelBoundary;
end

% Function for potentialBoundary
function pb = potentialBoundary(phiIntegral, r_out, theta_out)
    atol = 1e-08;
    lambda_value = 1.0;
    r_out = r_out(:);

    theta_out = theta_out(:);
    phiIntegral = phiIntegral(:);
    
    [R, TH, PH] = ndgrid(r_out, theta_out, phiIntegral);
    
    diffX = cos(PH) - cos(TH);
    diffY = sin(PH) - sin(TH);
    norm = sqrt(diffX.^2 + diffY.^2);
    
    lamSqrt = sqrt(lambda_value);
    bessel = -besselk(1, lamSqrt * norm);
    
    coeff = -lamSqrt / (4.0 * pi);
    kernel = coeff * bessel .* norm;
    
    eqMask = abs(PH - TH) < atol;
    kernel(eqMask) = 1.0 / (4.0 * pi);
    
    pb = kernel;
end

%% 
function val = laplace_solution_cartesian(x, y)
    val = 1- x.^2 - y.^2;
end

% Function to compute the Helmholtz solution in polar coordinates
function val = laplace_solution_polar(r, theta)
    x = r .* cos(theta);
    y = r .* sin(theta);
    val = laplace_solution_cartesian(x, y);
end
