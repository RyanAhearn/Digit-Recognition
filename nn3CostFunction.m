function [J grad] = nn3CostFunction(nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer1_size, ...
                                    hidden_layer2_size, ...
                                    num_labels, ...
                                    X, y, lambda)
%NN3COSTFUNCTION Implements the neural network cost function for a three layer
%neural network which performs classification
%   [J grad] = NN3COSTFUNCTON(nn_params, hidden_layer1_size, hidden_layer2_size, ...
%   num_labels, X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

end_theta2 = ((hidden_layer1_size * (input_layer_size + 1))) + ...
                    hidden_layer2_size * (hidden_layer1_size + 1);
                    
Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):end_theta2), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((1 + end_theta2):end), num_labels, (hidden_layer2_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% forward propigation
X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(size(z3, 1), 1) sigmoid(z3)];
z4 = a3 * Theta3';
h = sigmoid(z4);

% compute cost
if (num_labels > 1)
  y = eye(num_labels)(y, :);
endif

J = sum(sum((-y .* log(h)) .- ((1 .- y) .* log(1 .- h)))) / m;

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
t3 = Theta3(:, 2:end);

J += lambda / (2 * m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)) + sum(sum(t3 .^ 2)));

% compute gradient
err4 = h - y;
err3 = (err4*Theta3 .* sigmoidGradient([ones(size(z3, 1), 1) z3]))(:, 2:end);
err2 = (err3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);

Theta1_grad = err2' * X;
Theta2_grad = err3' * a2;
Theta3_grad = err4' * a3;

Theta1_grad = Theta1_grad / m + lambda * [zeros(hidden_layer1_size, 1) t1] / m;
Theta2_grad = Theta2_grad / m + lambda * [zeros(hidden_layer2_size, 1) t2] / m;
Theta3_grad = Theta3_grad / m + lambda * [zeros(num_labels, 1) t3] / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end