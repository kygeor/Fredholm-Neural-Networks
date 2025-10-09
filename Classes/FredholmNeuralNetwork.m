classdef FredholmNeuralNetwork < handle
    properties
        grid_dictionary   % Struct with fields 'layer_0', 'layer_1', ..., 'layer_K'
        kernel            % Function handle: kernel(in_value, out_value)
        additive          % Function handle: additive(value)
        initialization    % Function handle: initialization(value)
        grid_step         % Scalar: integration discretization step
        K                 % Number of integration layers (excluding layer_0)
        layer_sizes       % Vector: [input_size, numel(layer_0), numel(layer_1), ..., numel(layer_K)]
    end
    
    methods
        function obj = FredholmNeuralNetwork(grid_dictionary, kernel, additive, initialization, grid_step, K, input_size, output_size)
            % Constructor: store properties and compute layer sizes.
            obj.grid_dictionary = grid_dictionary;
            obj.kernel = kernel;
            obj.additive = additive;
            obj.initialization = initialization;
            obj.grid_step = grid_step;
            obj.K = K;
            % Compute layer sizes: first element is input_size, then use length of each grid.
            obj.layer_sizes = zeros(1, K + 2);
            obj.layer_sizes(1) = input_size;
            for i = 0:K
                field = sprintf('layer_%d', i);
                obj.layer_sizes(i + 2) = numel(grid_dictionary.(field));
            end
        end
        
        function [weights, biases] = compute_weights_and_biases(obj)
            % Precompute weights and biases while preserving differentiability.
            % We store weights and biases in cell arrays.
            weights = cell(obj.K + 2, 1);  % index 1 for layer 0, then 2..K+1 for subsequent layers.
            biases  = cell(obj.K + 2, 1);
            
            for i = 0:obj.K
                if i == 0
                    % For layer 0, create a diagonal matrix using the initialization function.
                    layer0 = obj.grid_dictionary.layer_0;
                    % Compute additive value via the initialization function.
                    additive_val = obj.initialization(layer0); 
                    % Create a diagonal matrix from the additive values.
                    W0 = diag(additive_val);
                    weights{1} = W0;
                    biases{1} = zeros(length(layer0), 1);
                else
                    % For layers i = 1,...,K:
                    % Get the previous and current grid.
                    grid_prev = obj.grid_dictionary.(sprintf('layer_%d', i - 1));
                    grid_curr = obj.grid_dictionary.(sprintf('layer_%d', i));
                    % Ensure grid_prev is a column vector and grid_curr is a row vector.
                    % grid_prev = grid_prev(:);
                    % grid_curr = grid_curr(:).';
                    grid_prev = grid_prev(:).';   % now 1 x n
                    grid_curr = grid_curr(:);      % now m x 1
                    % Compute the kernel matrix and multiply by the grid step.
                    weight_matrix = obj.kernel(grid_prev, grid_curr) * obj.grid_step;
                    weights{i+1} = weight_matrix;
                    
                    % Compute the bias vector using the additive function.
                    bias_val = obj.additive(obj.grid_dictionary.(sprintf('layer_%d', i)));
                    biases{i+1} = bias_val(:);
                end
            end
        end
        
        function [nn_output, layer_outputs] = forward(obj, predict_array)
            % Compute the forward pass.
            % First, precompute the weights and biases.
            [weights, biases] = obj.compute_weights_and_biases();
            
            % Initialize f_0 using the additive function on layer_0.
            init_val = obj.initialization(obj.grid_dictionary.layer_0);
            x = init_val(:);  % Ensure x is a column vector.
            layer_outputs = {x};
            
            % Propagate through layers 1 to K.
            for i = 1:obj.K
                % Multiply by the transposed weight matrix and add the bias.
                % x = weights{i+1}' * x + biases{i+1};
                x = weights{i+1} * x + biases{i+1};
                layer_outputs{end+1} = x;
            end
            % The final output.
            nn_output = x;
        end
    end
end
