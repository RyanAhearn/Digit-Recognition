function p = nn3Predict(Theta1, Theta2, Theta3, X)
%NN3PREDICT Predict the label of an input given a trained neural network
%   p = NN3PREDICT(Theta1, Theta2, Theta3, X) outputs the predicted label of X 
%   given the trained weights of a neural network (Theta1, Theta2, Theta3)
m = size(X, 1);

X = [ones(m, 1) X];
z2 = X * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = [ones(m, 1) sigmoid(z3)];
z4 = a3 * Theta3';
h = sigmoid(z4);

[dummy p] = max(h, [], 2);

end