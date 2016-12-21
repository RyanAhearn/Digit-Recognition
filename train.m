clear ; close all; clc


input_layer_size = 28 * 28;
hidden_layer1_size = 500;
hidden_layer2_size = 150;
num_labels = 10; % note: 0 is maped to 1, 1 to 2, ... 9 to 10

load('trainData.mat');

lambda = 3;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nn3CostFunction(p, ...
                                    input_layer_size, ...
                                    hidden_layer1_size, ...
                                    hidden_layer2_size, ...
                                    num_labels, X, y, lambda);
                                   
options = optimset('MaxIter', 100);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

end_theta2 = ((hidden_layer1_size * (input_layer_size + 1))) + ...
                    hidden_layer2_size * (hidden_layer1_size + 1);
                    
Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):end_theta2), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));               
                 
Theta3 = reshape(nn_params((1 + end_theta2):end), ...
                  num_labels, (hidden_layer2_size + 1));
                  

                 
save('-binary', 'weights.mat', 'Theta1', 'Theta2', 'Theta3');