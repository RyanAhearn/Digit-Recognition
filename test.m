clear ; close all; clc

load('weights.mat');
load('testData.mat');

p = nn3Predict(Theta1, Theta2, Theta3, Xtest);

fprintf('accuracy: %f\n', mean(ytest == p) * 100);