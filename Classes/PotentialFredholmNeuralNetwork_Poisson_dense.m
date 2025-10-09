classdef LimitInformedNeuralNetwork_PDE_dense
    properties
        fredholmModel            % FredholmNeuralNetwork_PDE instance
        diffPotentialsFn         % function handle @(phiGrid, r_out, theta_out)
        potentialBoundaryFn      % function handle @(phiGrid, r_out, theta_out)
        precomputedIntegralOut   % R×P array of precomputed integrals
        plotBIF logical = false  % whether to plot the base FNN output
    end

    methods
        function obj = LimitInformedNeuralNetwork_PDE_dense( ...
                fredholmModel, diffPotentialsFn, potentialBoundaryFn, ...
                precomputedIntegralOut, plotBIF)
            obj.fredholmModel          = fredholmModel;
            obj.diffPotentialsFn       = diffPotentialsFn;
            obj.potentialBoundaryFn    = potentialBoundaryFn;
            obj.precomputedIntegralOut = precomputedIntegralOut;
            if nargin>4
                obj.plotBIF = plotBIF;
            end
        end

        function y = forward(obj, input, r_out, theta_out, phiGrid, gridStep)
            % Forward pass inputs:
            %   input       P×1 input values
            %   r_out       R×1 radial coordinates for output points
            %   theta_out   P_out×1 angular coordinates for output points
            %   phiGrid     P×1 angular grid values for integration
            %   gridStep    scalar grid step size for integration
            
            % Get dimensions
            R = numel(r_out);
            P = numel(phiGrid);
            P_out = numel(theta_out);

            % Base Fredholm‐NN output (P×1)
            fnn_out = obj.fredholmModel.forward(phiGrid);

            % Optional plot of the base FNN
            if obj.plotBIF
                figure; 
                plot(phiGrid, fnn_out, 'LineWidth', 1.5);
                xlabel('Input'); ylabel('Output');
                title('Boundary function');
                grid on;
            end

            % ── Hidden "bias‐cancellation" layer ──%
            % Find for each theta_out the index in phiGrid (equivalent to argmin in PyTorch)
            thetaIdx = zeros(P_out, 1);
            for j = 1:P_out
                [~, thetaIdx(j)] = min(abs(phiGrid - theta_out(j)));
            end
            
            % Get the FNN output values at theta_out positions
            fnn_theta = fnn_out(thetaIdx);  % P_out×1
            
            % Expand fnn_theta to [R × P_out] for broadcasting
            fnn_theta_expanded = repmat(fnn_theta', [R, 1]);  % R×P_out

            % 1) Tile the FNN output across (r,θ_out) slices:
            fnn_out_expanded = reshape(fnn_out, [1, 1, P]);          % 1×1×P
            fnn_out_rep = repmat(fnn_out_expanded, [R, P_out, 1]);  % R×P_out×P

            hiddenOutput = fnn_out_rep - reshape(fnn_theta_expanded, [R, P_out, 1]); % R×P_out×P
            
            %── Output layer ──%
            % Get differential potentials and boundary potential terms
            W3 = obj.diffPotentialsFn(phiGrid, r_out, theta_out) * gridStep;  % R×P_out×P
            PB3 = obj.potentialBoundaryFn(phiGrid, r_out, theta_out);         % R×P_out×P
            
            % Verify dimensions
            assert(isequal(size(W3), [R, P_out, P]), ...
                'diffPotentials must be [R P_out P] but is [%d %d %d]', size(W3));
            assert(isequal(size(PB3), [R, P_out, P]), ...
                'potentialBoundary must be [R P_out P] but is [%d %d %d]', size(PB3));
            
            
            % Calculate first part of bias term:
            % torch.sum(fnn_output.unsqueeze(0).unsqueeze(0) * potential_boundary_term, dim=-1) * grid_step
            sumTerm = sum(fnn_out_rep .* PB3, 3) * gridStep;          % R×P_out

            % Calculate second part of bias term:
            % 0.5 * fnn_output_theta.unsqueeze(0).repeat(len(r_out), 1)
            halfTerm = 0.5 * fnn_theta_expanded;                        % R×P_out

            % Combine terms
            biasOut = sumTerm + halfTerm;                               % R×P_out
            
            % Add precomputed Poisson integral
            PI = obj.precomputedIntegralOut; 
            biasOut = biasOut + PI(:,thetaIdx);                                     % R×P_out

            % Perform weighted sum (equivalent to matmul in this case)
            weightedSum = sum(hiddenOutput.* W3, 3);          % R×P_out
            
            % Final output
            y = weightedSum + biasOut;                                  % R×P_out
        end
    end
end