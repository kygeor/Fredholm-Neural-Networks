classdef LimitInformedNeuralNetwork_PDE
    properties
        fredholmModel            % FredholmNeuralNetwork_PDE instance
        diffPotentialsFn         % function handle @(phiGrid, r_out, theta_out)
        potentialBoundaryFn      % function handle @(phiGrid, r_out, theta_out)
        precomputedIntegralOut   % R×P array of precomputed integrals
        plotBIF logical = false  % whether to plot the base FNN output
    end

    methods
        function obj = LimitInformedNeuralNetwork_PDE( ...
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
            fnn_out = obj.fredholmModel.forward(input);

            % Optional plot of the base FNN
            if obj.plotBIF
                figure; 
                plot(input, fnn_out, 'LineWidth', 1.5);
                xlabel('Input'); ylabel('Output');
                title('Boundary function');
                grid on;
            end

            %── Hidden "bias‐cancellation" layer ──%
            % Find for each theta_out the index in phiGrid (equivalent to argmin in PyTorch)
            thetaIdx = zeros(P_out, 1);
            for j = 1:P_out
                [~, thetaIdx(j)] = min(abs(phiGrid - theta_out(j)));
            end
            
            % Get the FNN output values at theta_out positions
            fnn_theta = fnn_out(thetaIdx);  % P_out×1
            
            % Expand fnn_theta to [R × P_out] for broadcasting
            fnn_theta_expanded = repmat(fnn_theta', [R, 1]);  % R×P_out
            
            % Create hidden bias (-fnn_theta_expanded)
            hiddenBias = -fnn_theta_expanded;  % R×P_out
            
            % Prepare fnn_out for broadcasting with hiddenBias
            % In PyTorch this was: fnn_output.unsqueeze(0).unsqueeze(0) + hidden_bias
            % Need to reshape for proper broadcasting with hiddenBias
            fnn_out_expanded = reshape(fnn_out, [1, 1, P]);  % 1×1×P
            
            % Apply the hidden bias to create hidden layer output
            % This creates a [R × P_out × P] tensor
            hiddenOutput = repmat(fnn_out_expanded, [R, P_out, 1]) + ...
                           reshape(hiddenBias, [R, P_out, 1]);  % R×P_out×P
            
            %── Output layer ──%
            % Get differential potentials and boundary potential terms
            W3 = obj.diffPotentialsFn(phiGrid, r_out, theta_out) * gridStep;  % R×P_out×P
            PB3 = obj.potentialBoundaryFn(phiGrid, r_out, theta_out);         % R×P_out×P
            
            % Verify dimensions
            assert(isequal(size(W3), [R, P_out, P]), ...
                'diffPotentials must be [R P_out P] but is [%d %d %d]', size(W3));
            assert(isequal(size(PB3), [R, P_out, P]), ...
                'potentialBoundary must be [R P_out P] but is [%d %d %d]', size(PB3));
            
            % Create expanded version of fnn_out for integral calculation
            fnn_out_expanded = reshape(fnn_out, [1, 1, P]);             % 1×1×P
            fnn_out_full = repmat(fnn_out_expanded, [R, P_out, 1]);     % R×P_out×P
            
            % Calculate first part of bias term:
            % torch.sum(fnn_output.unsqueeze(0).unsqueeze(0) * potential_boundary_term, dim=-1) * grid_step
            sumTerm = sum(fnn_out_full .* PB3, 3) * gridStep;          % R×P_out
            
            % Calculate second part of bias term:
            % 0.5 * fnn_output_theta.unsqueeze(0).repeat(len(r_out), 1)
            halfTerm = 0.5 * fnn_theta_expanded;                        % R×P_out
            
            % Combine terms
            biasOut = sumTerm + halfTerm;                               % R×P_out
            
            % Add precomputed Poisson integral
            PI = obj.precomputedIntegralOut;                            % R×P_out
            assert(isequal(size(PI), [R, P_out]), ...
                'precomputedIntegralOut must be [R P_out] but is [%d %d]', size(PI));
            biasOut = biasOut + PI;                                     % R×P_out
            
            % In PyTorch: torch.matmul(hidden_output.unsqueeze(-2), output_weights).squeeze(-2)
            % We need to do a weighted sum over the last dimension
            % Reshape W3 for proper broadcasting with hiddenOutput
            W3_reshaped = reshape(W3, [R, P_out, P, 1]);                % R×P_out×P×1
            hiddenOutput_reshaped = reshape(hiddenOutput, [R, P_out, P]);  % R×P_out×P
            
            % Perform weighted sum (equivalent to matmul in this case)
            weightedSum = sum(hiddenOutput_reshaped .* W3, 3);          % R×P_out
            
            % Final output
            y = weightedSum + biasOut;                                  % R×P_out
        end
    end
end