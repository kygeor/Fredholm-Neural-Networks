classdef LimitInformedNeuralNetwork_Helmholtz_dense
    properties
        fredholmModel                             % FredholmNeuralNetwork_PDE instance
        diffPotentialsFn                          % function handle @(phiGrid, r_out, theta_out)
        potentialBoundaryFn                       % function handle @(phiGrid, r_out, theta_out)
        precomputedIntegralEntireDomainSource     % R×P array of precomputed integrals (fundamental*source)
        precomputedIntegralEntireDomainFund       % R×P array of precomputed integrals (lambda*fundamental)
        lambdaValue                               % scalar lambda value
        plotBIF logical = false                   % whether to plot the base FNN output
    end

    methods
        function obj = LimitInformedNeuralNetwork_Helmholtz_dense( ...
                fredholmModel, diffPotentialsFn, potentialBoundaryFn, ...
                precomputedIntegralEntireDomainSource, precomputedIntegralEntireDomainFund, ...
                lambdaValue, plotBIF)
            obj.fredholmModel                         = fredholmModel;
            obj.diffPotentialsFn                      = diffPotentialsFn;
            obj.potentialBoundaryFn                   = potentialBoundaryFn;
            obj.precomputedIntegralEntireDomainSource = precomputedIntegralEntireDomainSource;
            obj.precomputedIntegralEntireDomainFund   = precomputedIntegralEntireDomainFund;
            obj.lambdaValue                           = lambdaValue;
            if nargin>6
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
            R     = numel(r_out);
            P     = numel(phiGrid);
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
            % Find for each theta_out the index in phiGrid
            thetaIdx = zeros(P_out,1);
            for j = 1:P_out
                [~, thetaIdx(j)] = min(abs(phiGrid - theta_out(j)));
            end

            % Get the FNN output values at theta_out positions
            fnn_theta = fnn_out(thetaIdx);             % P_out×1
            fnn_theta_expanded = repmat(fnn_theta', [R, 1]);  % R×P_out

            % Tile the FNN output across (r,θ_out) slices
            fnn_out_expanded = reshape(fnn_out, [1,1,P]);        % 1×1×P
            fnn_out_rep      = repmat(fnn_out_expanded, [R, P_out, 1]);  % R×P_out×P

            % Hidden layer
            hiddenOutput = fnn_out_rep - reshape(fnn_theta_expanded, [R, P_out, 1]);  % R×P_out×P

            %── Output layer ──%
            % Differential kernels and boundary kernels
            W3  = obj.diffPotentialsFn(phiGrid, r_out, theta_out) * gridStep;  % R×P_out×P
            PB3 = obj.potentialBoundaryFn(phiGrid, r_out, theta_out);         % R×P_out×P

            assert(isequal(size(W3),  [R, P_out, P]), ...
                'diffPotentials must be [R P_out P] but is [%d %d %d]', size(W3));
            assert(isequal(size(PB3), [R, P_out, P]), ...
                'potentialBoundary must be [R P_out P] but is [%d %d %d]', size(PB3));

            % 1) Boundary‐integral term
            sumTerm = sum(fnn_out_rep .* PB3, 3) * gridStep;  % R×P_out

            % 2) Add precomputed fundamental*source integral (subsample at theta_out)
            intSource = obj.precomputedIntegralEntireDomainSource(:, thetaIdx);  % R×P_out
            biasOutput = sumTerm + intSource;                                     % R×P_out

            % 3) Add lambda*fundamental correction term
            intFund = obj.precomputedIntegralEntireDomainFund(:, thetaIdx);       % R×P_out
            boundaryVals     = intFund(end, :);                                   % 1×P_out
            sqrtLambdaDiff   = intFund - repmat(boundaryVals, [R, 1]);            % R×P_out
            factor           = 0.5 + sqrtLambdaDiff;                              % R×P_out
            halfTermCorrected = factor .* fnn_theta_expanded;                     % R×P_out
            biasOutput       = biasOutput + halfTermCorrected;                    % R×P_out

            % Weighted‐sum (integral) of the hidden layer
            weightedSum = sum(hiddenOutput .* W3, 3);                             % R×P_out

            % Final solution
            y = weightedSum + biasOutput;                                         % R×P_out
        end
    end
end
