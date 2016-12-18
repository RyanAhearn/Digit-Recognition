function p = nn3Predict(Theta1, Theta2, Theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, Theta3, X) outputs the predicted label of X 
%   given the trained weights of a neural network (Theta1, Theta2, Theta3)

X = [ones(size(X, 1), 1) X];
z2 = X * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(size(z2, 1), 1) sigmoid(z3)];
z4 = a3 * Theta3';
h = sigmoid(z4);

[dummy p] = max(h, [], 2);

end