classdef FredholmNeuralNetwork_PDE_dense
    properties
        gridDictionary           % struct with fields layer_0, layer_1, …, layer_K
        kernel                   % function handle @(inVal,outVal)
        gridStep   double        % scalar step size (Δφ)
        K          double        % number of layers
        inputSize  double        % size of input vector
        outputSize double        % size of output vector
        kmConstant double        % km constant
        funcFn                   % function handle @(θ)
        precomputedIntegralMap   % containers.Map of θ→integral
    end
    
    methods
        function obj = FredholmNeuralNetwork_PDE_dense(gridDictionary, kernel, gridStep, K, inputSize, outputSize, kmConstant, funcFn, precomputedIntegralMap)
            % Constructor: store all parameters
            obj.gridDictionary         = gridDictionary;
            obj.kernel                 = kernel;
            obj.gridStep               = gridStep;
            obj.K                      = K;
            obj.inputSize              = inputSize;
            obj.outputSize             = outputSize;
            obj.kmConstant             = kmConstant;
            obj.funcFn                 = funcFn;
            obj.precomputedIntegralMap = precomputedIntegralMap;
        end
        
        function additiveVals = additive(obj, outValues)
            % Compute additive term: 2*(funcFn(θ) - precomputedIntegral(θ))
            v       = outValues(:);
            keysArr = cell2mat(obj.precomputedIntegralMap.keys).';
            valsArr = cell2mat(obj.precomputedIntegralMap.values).';
            additiveVals = zeros(size(v));
            for j = 1:numel(v)
                % find the index of the matching θ
                idx = find(keysArr == v(j), 1, 'first');
                additiveVals(j) = 2*( obj.funcFn(v(j)) - valsArr(idx) );
            end
            additiveVals = reshape(additiveVals, size(outValues));
            additiveVals = additiveVals(:);
        end
        
        function [Wcell, bcell] = computeWeightsAndBiases(obj)
            % Precompute weight matrices and bias vectors for each layer
            Wcell = cell(1, obj.K+1);
            bcell = cell(1, obj.K+1);
            for i = 0:obj.K
                if i == 0
                    % Input layer: identity weights, zero bias
                    Wcell{1} = eye(obj.inputSize);
                    bcell{1} = zeros(obj.inputSize,1);
                else
                    prev = obj.gridDictionary.(sprintf('layer_%d', i-1));
                    curr = obj.gridDictionary.(sprintf('layer_%d', i));
                    prev_col = prev(:).';
                    curr_row = curr(:);
                    % Kernel-based weight matrix
                    Kmat = obj.kernel(prev_col, curr_row);
                    W    = Kmat * obj.gridStep * obj.kmConstant + ...
                           (1 - obj.kmConstant)*(prev_col == curr_row);
                    Wcell{i+1} = W;
                    % Bias from the double-layer additive term
                    bcell{i+1} = obj.additive(curr) * obj.kmConstant;
                end
            end
        end
        
        function y = forward(obj, predict_array)
            % FORWARD  Evaluate the Fredholm NN at arbitrary query angles.
            %
            %   y = forward(obj, predict_array)
            %
            % Input:
            %   predict_array : P×1 (or 1×P) vector of θ values to predict at
            %
            % Output:
            %   y : P×1 vector of network output u at those θ's

            % 1) Compute the "hidden‐layer" output on layer_K grid
            [Wcell, bcell] = obj.computeWeightsAndBiases();
            x = ones(obj.inputSize,1);
            for i = 1:numel(Wcell)
                x = Wcell{i} * x + bcell{i};
            end
            nn_output = x;  % N×1, N = numel(gridDictionary.layer_K)
            y = nn_output;

            % % 2) Extract the layer_K grid points (φ values)
            % grid_K = obj.gridDictionary.(sprintf('layer_%d', obj.K));
            % grid_K = grid_K(:).';  % 1×N row
            % 
            % % 3) For each θ in predict_array, form final kernel‐vector and bias
            % P = numel(predict_array);
            % y = zeros(P,1);
            % for j = 1:P
            %     th = predict_array(j);
            %     % kernel vector between grid_K and th
            %     kvec = obj.kernel(grid_K, th);      % 1×N
            %     wvec = kvec * obj.gridStep;         % 1×N
            %     b    = obj.additive(th);            % scalar
            %     % dot‐product + bias
            %     y(j) = wvec * nn_output + b;
            % end
        end
    end
end
