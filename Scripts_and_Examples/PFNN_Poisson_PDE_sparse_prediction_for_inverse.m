% Script that solves the Poisson PDE over a sparse grid to produce data used for the inverse %source term problem

% Parameters
K = 100;
phi_0     = 0;
phi_n     = 2*pi;
phi_steps = 100;
dphi      = (phi_n - phi_0)/phi_steps;
phi       = phi_0 : dphi : (phi_n - dphi);

% Subsample it to get 20 boundary angles exactly on that grid:
n = round(phi_steps/20);                
theta_out = phi(1:n:end);       

phi_grid_dict = struct();
for i = 0:K
    fld = sprintf('layer_%d', i);
    phi_grid_dict.(fld) = phi;
end

% Boundary grid for output
r_out     = linspace(0,1,20);

% Precompute integrals (Fredholm additive term on boundary r_out=1)
tic;
precomputed_integrals     = parallelIntegrateMeshgrid(r_out, phi);
precomputed_integrals_BIE = precomputed_integrals(end, :);
integral_time = toc;
fprintf('Integral computation time: %.6f seconds\n', integral_time);

% Build lookup map for theta_out → integral
keys   = num2cell(phi);
values = num2cell(precomputed_integrals_BIE);
precomputed_integral_map = containers.Map(keys, values);

input_size = numel(phi);
output_size = numel(phi);

%%
% Model parameters
km_constant = 0.3;  % km constant

% Instantiate Fredholm model
fredholm_model = FredholmNeuralNetwork_PDE_dense(...
    phi_grid_dict, ...                % grid dictionary
    @kernel, ...                      % kernel handle
    dphi, ...                         % grid step
    K, ...                            % number of layers
    input_size, ...                   % input size
    output_size, ...
    km_constant, ...                  % km constant
    @funcFn, ...                      % boundary function handle
    precomputed_integral_map ...      % map of precomputed integrals
    );

% Instantiate Limit‑Informed model
tic;
limit_model = PotentialFredholmNeuralNetwork_Poisson_dense(...
    fredholm_model, ...
    @diffPotentialsLimit, ...
    @potentialBoundary, ...
    precomputed_integrals, ...
    true);
limit_init_time = toc;
fprintf('Model initialization time: %.6f seconds\n', limit_init_time);

% Forward pass on the full pipeline
tic;
output = limit_model.forward(theta_out, r_out, theta_out, phi, dphi);  % R×P
forward_time = toc;
fprintf('Forward pass time: %.6f seconds\n', forward_time);

%%
% Set LaTeX-compatible default settings
set(groot, ...
    'defaultAxesFontName', 'Times New Roman', ...
    'defaultTextFontName', 'Times New Roman', ...
    'defaultLegendFontName', 'Times New Roman', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...
    'defaultTextInterpreter', 'latex', ...
    'defaultLegendInterpreter', 'latex', ...
    'defaultAxesFontSize', 26, ...
    'defaultTextFontSize', 26, ...
    'defaultLegendFontSize', 26);

% Visualization: contour in (theta, r) space
% Visualization: contour in (theta, r) space
[R_grid, Theta_grid] = ndgrid(r_out, theta_out);
figure('Position',[100,100,500,400]);
contourf(Theta_grid, R_grid, output, 50, 'LineColor','none');

cb = colorbar;
cb.Label.String       = '$u(r, \phi)$';   % LaTeX formatted label
cb.Label.Interpreter  = 'latex';

% Increase colorbar font sizes
cb.Label.FontSize     = 26;   % label size
cb.FontSize           = 26;   % tick labels size

xlabel('$\phi$', 'Interpreter', 'latex');  
ylabel('$r$',    'Interpreter', 'latex');

ytickformat('%.1f');
xtickformat('%.1f');

ax = gca;
set(ax, 'FontSize', 26, 'TickLabelInterpreter', 'latex');
axis tight;
axis square;

set(gcf,'PaperPositionMode','auto');

% Export a tight, vector PDF
exportgraphics(gcf, 'output_laplace_v2.pdf', ...
               'ContentType','vector');




%%
% 3D surface plot of the results
figure('Position',[100,100,500,400]);
surf(Theta_grid, R_grid, output, 'EdgeColor','none');
xlabel('theta'); ylabel('r'); zlabel('u(r,	heta)');
title('3D Surface of Limit‑Informed PDE Output');
set(gca,'FontSize',26);
view(45,30);
colorbar;

%% Plot differences to known solution.

% Contour plot for visualization of absolute error in (r, theta) space
difference = abs(output - laplace_solution_polar(R_grid, Theta_grid));

figure('Position',[100,100,500,400]);
contourf(Theta_grid, R_grid, difference, 50, 'LineColor','none');

cb = colorbar;
cb.Label.String       = '$|\tilde{u}(r,\phi) - u(r,\phi)|$';  % LaTeX formatted label
cb.Label.Interpreter  = 'latex';
% Increase colorbar font sizes
cb.Label.FontSize     = 26;   % colorbar label
cb.FontSize           = 26;   % colorbar tick labels

xlabel('$\phi$', 'Interpreter', 'latex', 'FontSize', 26);
ylabel('$r$',    'Interpreter', 'latex', 'FontSize', 26);

ytickformat('%.1f');
xtickformat('%.1f');

ax = gca;
set(ax, ...
    'FontSize', 26, ...                  % axis tick labels
    'TickLabelInterpreter', 'latex' ...  % ensure LaTeX ticks
);
axis tight;
axis square;

set(gcf,'PaperPositionMode','auto');

% 2) Export a tight, vector PDF
exportgraphics(gcf, 'error_laplace_v2.pdf', ...
               'ContentType','vector');



%% Local functions

% Integration functions and kernel

function results = parallelIntegrateMeshgrid(r_out_vals, theta_out_vals)
    tol = 1e-8;
    [R_out, Theta_out] = ndgrid(r_out_vals, theta_out_vals);
    Rf = R_out(:); Tf = Theta_out(:);
    results_flat = zeros(numel(Rf),1);
    parfor i = 1:numel(Rf)
        results_flat(i) = integrateForOutValueScalar(Rf(i), Tf(i), tol);
    end
    results = reshape(results_flat, size(R_out));
end

function val = integrateForOutValueScalar(r_out, theta_out, tol)
    % Outer integral over r2
    f_r2 = @(r2) innerIntegration(r2, r_out, theta_out, tol);
    val  = integral(f_r2, 0, 1, 'AbsTol', tol, 'RelTol', tol, ...
                    'Waypoints', r_out, 'ArrayValued', true);
end

function inner = innerIntegration(r2, r_out, theta_out, tol)
    % Inner integral over theta
    integrand_theta = @(th) fundamentalFn(r2, th, r_out, theta_out) .* sourceTerm(r2, th);
    if abs(r2 - r_out) < tol
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, ...
                         'Waypoints', theta_out, 'ArrayValued', true);
    else
        inner = integral(integrand_theta, 0, 2*pi, 'AbsTol', tol, 'RelTol', tol, ...
                         'ArrayValued', true);
    end
end

function fund = fundamentalFn(r2, theta, r_out, theta_out)
    % Compute the 2D fundamental solution * r2, vectorized in theta
    % Inputs: r2 (scalar), theta (vector), r_out (scalar), theta_out (scalar)
    % Compute Cartesian differences correctly:
    dx = r_out*cos(theta_out) - r2*cos(theta);
    dy = r_out*sin(theta_out) - r2*sin(theta);

    R2 = dx.^2 + dy.^2;            % squared distance
    fund = (1/(2*pi)) * 0.5 ...
           * log(R2) .* r2;         % matches the Python 1/(2π)·½·log(...)·r2
end


function src = sourceTerm(r2, th)
    % For the Laplace PDE the source term is zero everywhere
    % r2 and th may be vectors, so return the same size
    src = zeros(size(th));
end

function k = kernel(inValue, outValue)
    weight = 1/(4*pi);
    k = -2 .* weight .* ones(size(inValue - outValue));
end

function out = funcFn(outValue)
    % New Dirichlet boundary condition: cos^2(theta) - sin^2(theta) + 1
    out = cos(outValue).^2 - sin(outValue).^2 + 1;
end

% Potential and LINN functions

function kernelLimit = diffPotentialsLimit(phiIntegral, r_out, theta_out)
% DIFFPOTENTIALSLIMIT  3-D broadcasted "limit" kernel for LINN
%   kernelLimit = diffPotentialsLimit(phiIntegral, r_out, theta_out)
%
%  Inputs:
%    phiIntegral : 1×P vector of quadrature angles
%    r_out       : 1×N or N×1 vector of boundary radii
%    theta_out   : 1×M or M×1 vector of boundary angles
%
%  Output:
%    kernelLimit : N×M×P array of the kernel limit values

  % Ensure column vectors
  r   = r_out(:);
  th  = theta_out(:);
  ph  = phiIntegral(:);

  % Build 3-D grid (N×M×P)
  [R, TH, PH] = ndgrid(r, th, ph);

  % Boundary mask (where r_out == 1)
  mask = (R == 1.0);

  % Numerator & denominator of the integral kernel
  num = cos(PH) .* (cos(PH) - R .* cos(TH)) + ...
        sin(PH) .* (sin(PH) - R .* sin(TH));
  den = (cos(PH) - R .* cos(TH)).^2 + ...
        (sin(PH) - R .* sin(TH)).^2;

  Kmat = num ./ den;

  % Enforce the half-value on the boundary
  Kmat(mask) = 0.5;

  % Final "limit" kernel
  kernelLimit = (1 / (2*pi)) * (Kmat - 0.5);
end


function pb = potentialBoundary(phiIntegral, r_out, theta_out)
% POTENTIALBOUNDARY  Constant boundary potential for LINN
%   pb = potentialBoundary(phiIntegral, r_out, theta_out)
%
%  Inputs:
%    phiIntegral : 1×P vector of quadrature angles
%    r_out       : 1×N or N×1 vector of boundary radii (ignored)
%    theta_out   : 1×M or M×1 vector of boundary angles (ignored)
%
%  Output:
%    pb : N×M×P array all filled with the constant 1/(4π)

  % Ensure vectors
  r   = r_out(:);
  th  = theta_out(:);
  ph  = phiIntegral(:);

  % Build a dummy grid
  [R,~,~] = ndgrid(r, th, ph);

  % Constant value 1/(2π)*0.5 = 1/(4π)
  constantVal = 1 / (4*pi);

  % Fill entire 3-D array
  pb = constantVal * ones(size(R));
end

%% True solution functions
% Function to compute the Laplace solution in Cartesian coordinates
% function val = laplace_solution_cartesian(x, y)
%     val = (1/4) * (x.^2 + y.^2 - 1) .* x;
% end
% 
% % Function to compute the Laplace solution in polar coordinates
% function val = laplace_solution_polar(r, theta)
%     x = r .* cos(theta);
%     y = r .* sin(theta);
%     val = laplace_solution_cartesian(x, y);
% end

function val = laplace_solution_polar(r, theta)
    % Analytic solution in polar for u(x,y) = x^2 - y^2 + 1
    % where x = r cos(theta), y = r sin(theta)
    x = r .* cos(theta);
    y = r .* sin(theta);
    val = x.^2 - y.^2 + 1;
end
