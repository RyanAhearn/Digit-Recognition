image_size = 28 * 28;
num_training_examples = 60000;
num_test_examples = 10000;

fid = fopen('train-images.idx3-ubyte');
X = fread(fid)(17:end);
fclose(fid);

X = reshape(X, image_size, num_training_examples)';
for i = 1:num_training_examples
  X(i, :) = reshape(X(i, :), 28, 28)'(:);
endfor

fid = fopen('train-labels.idx1-ubyte');
y = fread(fid)(9:end) .+ 1;
fclose(fid);

fid = fopen('t10k-images.idx3-ubyte');
Xtest = fread(fid)(17:end);
fclose(fid);

Xtest = reshape(Xtest, image_size, num_test_examples)';
for i = 1:num_test_examples
  Xtest(i, :) = reshape(Xtest(i, :), 28, 28)'(:);
endfor

fid = fopen('t10k-labels.idx1-ubyte');
ytest = fread(fid)(9:end) .+ 1;
fclose(fid);

save('-binary', 'trainData.mat', 'X', 'y');
save('-binary', 'testData.mat', 'Xtest', 'ytest');