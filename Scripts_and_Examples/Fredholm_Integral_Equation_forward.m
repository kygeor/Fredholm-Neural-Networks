% This script reproduces the PyTorch code for running the FredholmNeuralNetwork. It is an example of the solution %to the forward Fredholm Integral Equation

% Define lambda.
lamb = 0.3;

% Define the kernel function.
% It accepts an input vector (column) and an output vector (row) and computes:
%   lamb * (cos(25*(out_value - in_value)) + cos(7*(out_value - in_value)))
kernel = @(in_value, out_value) lamb * (cos(25 * (out_value - in_value)) + cos(7 * (out_value - in_value)));

% Define the additive function.
additive = @(value) sin(25 * value) + sin(7 * value);

% Set example parameters.
K = 10;            % Number of layers (excluding layer_0)
N = 300;          % Number of grid points
input_size = 300; % Input layer size
output_size = 300; % Output layer size (not explicitly used in this example)

% Create a grid with N points in the interval [0, 1]
y = linspace(0, 1, N);

% Create grid_dictionary as a struct with fields 'layer_0', 'layer_1', ..., 'layer_K'
grid_dictionary = struct();
for i = 0:K
    field_name = sprintf('layer_%d', i);
    grid_dictionary.(field_name) = y;
end

% Compute the grid step.
dy = (y(end) - y(1)) / (N - 1);

% Create a predict array (here, simply the grid).
predict_array = y;

% Instantiate the network.
% The constructor is assumed to have the signature:
% FredholmNeuralNetwork(grid_dictionary, kernel, additive, initialization, grid_step, K, input_size, output_size)
% In this case, we pass additive twice (once for additive and once for initialization).
model = FredholmNeuralNetwork(grid_dictionary, kernel, additive, additive, dy, K, input_size, output_size);

% Run the forward pass. The network returns the final prediction and the outputs at each layer.
[final_prediction, layer_outputs] = model.forward(predict_array);

% Display the shape (size) of outputs from each layer.
for i = 1:length(layer_outputs)
    layer_output = layer_outputs{i};
    fprintf('Layer %d output shape: %s\n', i-1, mat2str(size(layer_output)));
end

%% Plot the final prediction.
figure;
plot(y, final_prediction, 'LineWidth', 2);
xlabel('y');
ylabel('Final Prediction');
title('Final Prediction from Fredholm Neural Network');
grid on;
