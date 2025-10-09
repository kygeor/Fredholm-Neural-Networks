classdef LimitInformedNeuralNetwork_Helmholtz
    properties
        fredholmModel            % FredholmNeuralNetwork_PDE instance
        diffPotentialsFn         % function handle @(phiGrid, r_out, theta_out)
        potentialBoundaryFn      % function handle @(phiGrid, r_out, theta_out)
        precomputedIntegralEntireDomainSource   % R×P array of precomputed integrals (fundamental*source)
        precomputedIntegralEntireDomainFund     % R×P array of precomputed integrals (lambda*fundamental)
        lambdaValue              % scalar lambda value
        plotBIF logical = false  % whether to plot the base FNN output
    end

    methods
        function obj = LimitInformedNeuralNetwork_Helmholtz( ...
                fredholmModel, diffPotentialsFn, potentialBoundaryFn, ...
                precomputedIntegralEntireDomainSource, precomputedIntegralEntireDomainFund, lambdaValue, plotBIF)
            obj.fredholmModel          = fredholmModel;
            obj.diffPotentialsFn       = diffPotentialsFn;
            obj.potentialBoundaryFn    = potentialBoundaryFn;
            obj.precomputedIntegralEntireDomainSource = precomputedIntegralEntireDomainSource;
            obj.precomputedIntegralEntireDomainFund = precomputedIntegralEntireDomainFund;
            obj.lambdaValue            = lambdaValue;
            if nargin > 6
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
            
            % Expand fnn_theta to [R × P_out × 1] for broadcasting
            fnn_theta_expanded = repmat(fnn_theta', [R, 1, 1]);  % R×P_out×1
            
            % Create hidden bias (-fnn_theta_expanded)
            hiddenBias = -fnn_theta_expanded;  % R×P_out×1
            
            % Prepare fnn_out for broadcasting with hiddenBias
            fnn_out_expanded = reshape(fnn_out, [1, 1, P]);  % 1×1×P
            
            % Apply the hidden bias to create hidden layer output
            hiddenOutput = repmat(fnn_out_expanded, [R, P_out, 1]) + hiddenBias;  % R×P_out×P
            
            %── Output layer ──%
            % Get differential potentials and boundary potential terms
            W3 = obj.diffPotentialsFn(phiGrid, r_out, theta_out) * gridStep;  % R×P_out×P
            PB3 = obj.potentialBoundaryFn(phiGrid, r_out, theta_out);         % R×P_out×P
            
            % Compute the bias output term
            biasOutput = sum(fnn_out_expanded .* PB3, 3) * gridStep;  % R×P_out
            
            % Add the precomputed integral (fundamental*source)
            integralFundSource = obj.precomputedIntegralEntireDomainSource;  % R×P_out
            biasOutput = biasOutput + integralFundSource;
            
            % Compute the factor term
            integralSqrtLambdaFund = obj.precomputedIntegralEntireDomainFund;  % R×P_out
            boundaryVals = integralSqrtLambdaFund(end, :);  % 1×P_out
            sqrtLambdaDiff = integralSqrtLambdaFund - repmat(boundaryVals, [R, 1]);  % R×P_out
            factor = 0.5 + sqrtLambdaDiff;  % R×P_out
            
            % Multiply factor by fnn_theta_expanded and add to biasOutput
            halfTermCorrected = factor .* fnn_theta_expanded(:, :, 1);  % R×P_out
            biasOutput = biasOutput + halfTermCorrected;
            
            % Reshape W3 for proper broadcasting with hiddenOutput
            W3_reshaped = reshape(W3, [R, P_out, P, 1]);  % R×P_out×P×1
            
            % Perform weighted sum (equivalent to matmul)
            weightedSum = sum(hiddenOutput .* W3_reshaped, 3);  % R×P_out
            
            % Final output
            y = weightedSum + biasOutput;  % R×P_out
        end
    end
end