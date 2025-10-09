% Script for PFNN to solve the Helmholtz PDE. The script uses the simple PFNN implementation    %(instead of the "dense" version).

% Parameters
K = 100;
phi_grid_dict = struct();
for i = 0:K
    phi_0 = 0.0;
    phi_n = 2.0 * pi;
    phi_steps = 1000;
    dphi = (phi_n - phi_0) / phi_steps;
    phi = phi_0:dphi:phi_n-dphi;
    phi_grid_dict.(sprintf('layer_%d', i)) = phi;
end

% Define the grid for prediction (r is needed for the LINN)
r_out = linspace(0, 1, 100);
theta_out = phi;

% Precompute integrals (Fredholm additive term on boundary r_out=1)
tic;
% Precompute integrals for the output values
precomputed_integrals = parallelIntegrateMeshgrid(r_out, theta_out);
precomputed_integrals_source = precomputed_integrals(:, :, 1);
precomputed_integrals_lambda = precomputed_integrals(:, :, 2);
precomputed_integrals_BIE = precomputed_integrals_source(end, :);
precomputed_integrals_lambda_boundary = precomputed_integrals_lambda(end, :);
% Convert results to map for quick lookup
precomputed_integral_map = containers.Map(num2cell(theta_out), num2cell(precomputed_integrals_BIE));
integral_time = toc;
fprintf('Integral computation time: %.6f seconds\n', integral_time);


%% Model parameters
% Model parameters
input_size = numel(phi);
output_size = numel(theta_out);
km_constant = 0.3;
lambda_value = 1.0;

% Instantiate the model with precomputed integrals
fredholm_model = FredholmNeuralNetwork_PDE(phi_grid_dict, @kernel, dphi, K, input_size, output_size, km_constant, @funcFn, precomputed_integral_map);

% % Forward pass
% output = fredholm_model.forward(theta_out');
% 
% % Plot the output
% figure;
% plot(theta_out, output, 'LineWidth', 1.5);
% xlabel('Predict Array (x_list)');
% ylabel('Output');
% title('Model Output vs. Predict Array');
% legend('Model Output');
% grid on;

% Instantiate the LimitInformed model with the precomputed integrals
limit_informed_model = PotentialFredholmNeuralNetwork_Helmholtz(fredholm_model, @diffPotentialsLimit, @potentialBoundary, precomputed_integrals_source, precomputed_integrals_lambda, lambda_value, false);

% Forward pass
tic;
output = limit_informed_model.forward(theta_out', r_out, theta_out, phi, dphi);
forward_pass_time = toc;
fprintf('Time to forward pass: %.3f s\n', forward_pass_time);


% Contour plot for visualization in (r, theta) space
[R_out_grid, Theta_out_grid] = ndgrid(r_out, theta_out);
figure;
contourf(Theta_out_grid, R_out_grid, output, 50, 'LineColor', 'none');
colorbar;
xlabel('\theta', 'FontSize', 18);
ylabel('r', 'FontSize', 18);
title('$\tilde{u}(r,\phi)$', 'Interpreter', 'latex', 'FontSize', 18);
set(gca, 'FontSize', 16);



% Visualization: contour in (theta, r) space
[R_grid, Theta_grid] = ndgrid(r_out, theta_out);
figure('Position',[100,100,800,600]);
contourf(Theta_grid, R_grid, output, 50, 'LineColor','none');
colorbar;
xlabel('\theta');
ylabel('r');
title('Limit‑Informed PDE Output');
set(gca,'FontSize',14);

%%
% 3D surface plot of the results
figure('Position',[200,200,800,600]);
surf(Theta_grid, R_grid, output, 'EdgeColor','none');
xlabel('theta'); ylabel('r'); zlabel('u(r,	heta)');
title('3D Surface of Limit‑Informed PDE Output');
set(gca,'FontSize',14);
view(45,30);
colorbar;

%% Plot differences to known solution.

% Contour plot for visualization of absolute error in (r, theta) space
[R_grid, Theta_grid] = ndgrid(r_out, theta_out);
figure('Position',[300,300,800,600]);
difference = abs(output - laplace_solution_polar(R_grid, Theta_grid));
contourf(Theta_grid, R_grid, difference, 50, 'LineColor','none');
colorbar;
xlabel('\phi', 'FontSize', 18);
ylabel('r', 'FontSize', 18);
title('Absolute Error: $|\tilde{u}(r,\phi) - u(r,\phi)|$', 'Interpreter', 'latex', 'FontSize', 18);
set(gca, 'FontSize', 16);



%% Local functions

% Integration functions and kernel for Helmholtz PDE

% Global parameter
lambda_value = 1.0;

% Tolerance for deciding if r2 is "close" to r_out

function results = parallelIntegrateMeshgrid(r_out_vals, theta_out_vals)
    tol = 1e-8;
    [R_out, Theta_out] = ndgrid(r_out_vals, theta_out_vals);
    Rf = R_out(:); Tf = Theta_out(:);
    results_flat_1 = zeros(numel(Rf), 1);
    results_flat_2 = zeros(numel(Rf), 1);
    
    parfor i = 1:numel(Rf)
        [result1, result2] = integrateForOutValueScalar(Rf(i), Tf(i), tol);
        results_flat_1(i) = result1;
        results_flat_2(i) = result2;
    end
    
    results_array_1 = reshape(results_flat_1, size(R_out));
    results_array_2 = reshape(results_flat_2, size(R_out));
    results = cat(3, results_array_1, results_array_2);
end

function [result1, result2] = integrateForOutValueScalar(r_out, theta_out, tol)
    % Outer integral over r2
    f_r2_1 = @(r2) innerIntegration(r2, r_out, theta_out, tol, @integrandScalar);
    f_r2_2 = @(r2) innerIntegration(r2, r_out, theta_out, tol, @integrandScalarBoundary);
    
    result1 = integral(f_r2_1, 0, 1, 'AbsTol', tol, 'RelTol', tol, 'Waypoints', r_out, 'ArrayValued', true);
    result2 = integral(f_r2_2, 0, 1, 'AbsTol', tol, 'RelTol', tol, 'Waypoints', r_out, 'ArrayValued', true);
end

function inner = innerIntegration(r2, r_out, theta_out, tol, integrand_fn)
    % Inner integral over theta
    integrand_theta = @(th) integrand_fn(r2, th, r_out, theta_out);
    if abs(r2 - r_out) < tol
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, 'Waypoints', theta_out, 'ArrayValued', true);
    else
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, 'ArrayValued', true);
    end
end

function val = integrandScalar(r_2, theta, r_out, theta_out)
    fundamental = fundamentalFnScalar(r_2, theta, r_out, theta_out);
    source_term = sourceTermFnScalar(r_2, theta);
    val = source_term .* fundamental .* r_2;
end

function val = integrandScalarBoundary(r_2, theta, r_out, theta_out)
    lambda_value = 1.0;
    fundamental = fundamentalFnScalar(r_2, theta, r_out, theta_out);
    val = sqrt(lambda_value) .* fundamental .* r_2;
end

function term = fundamentalFnScalar(r_2, theta, r_out, theta_out)
    lambda_value = 1.0;
    norm = sqrt((r_out*cos(theta_out) - r_2*cos(theta)).^2 + (r_out*sin(theta_out) - r_2*sin(theta)).^2);
    term = (-1/(2*pi)) * besselk(0, sqrt(lambda_value) * norm);
end

function src = sourceTermFnScalar(r_2, theta)
    src = -(r_2.*cos(theta)).^3 + 6*r_2.*cos(theta) + 2*(r_2.*sin(theta)).^2 - 4;
end

%% Functions for Helmholtz case

% Function for func_fn
function func = funcFn(outValue)
    func = (cos(outValue)).^3 + 2*(cos(outValue)).^2 - 2;
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

% Function for diffPotentialsLimit
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

%% True solution functions
% Function to compute the Helmholtz solution in Cartesian coordinates
function val = laplace_solution_cartesian(x, y)
    val = x.^3 - 2*(y.^2);
end

% Function to compute the Helmholtz solution in polar coordinates
function val = laplace_solution_polar(r, theta)
    x = r .* cos(theta);
    y = r .* sin(theta);
    val = laplace_solution_cartesian(x, y);
end
