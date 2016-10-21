function data = generateData(mean, variance, N)

sz = max(size(mean));
variance = variance * eye(sz, sz);
data = mvnrnd(mean, variance, N);


Nh = 10; % Number of hidden layer neurons
Epoch = 150; % Number of Epochs
eta = 0.1;
alpha = [0, 0.4, 0.7, 0.9];

W10(:, 1) = 2 * rand(Nh,1) - 1;
W10(:, 2) = 2 * rand(Nh,1) - 1;
W21 = 2 * rand(Nh,1) - 1;
dW21 = 0;
dW10 = 0;