%% Generating Training Samples
clc;clear
Data1 = generateData([0,0],1,250);
Data2 = generateData([2,0],2,250);
Xtrain = [Data1;Data2];
d(1:250,1) = 1; % Desired Value for Output
d(251:500,1) = -1;
% Implementing The Network
Nh = 10; % Number of hidden layer neurons
Epoch = 10; % Number of Epochs
eta1 = 0.1;
eta2 = 0.1;
alpha = 0;
% initializing Weights in [-1,1]
W10 = 2 * rand(Nh,2) - 1; % Input -> Hidden Layer
W21 = 2 * rand(Nh,1) - 1; % Hidden Layer -> Output Layer
dW21 = 0;
dW10 = 0;
gamma = 0.01;
deta2 = 0;
delta2 = 1;
deta1 = 0;
delta1 = 1;
% Training the Network
for ep = 1:Epoch % Epochs loop
  for n = 1:size(Xtrain,1) % Training Data loop
    vh1 = W10 * Xtrain(n,:)';
    yh1 = tanh(vh1./2); % Hidden Layer Output
    vo2 = W21' * yh1;
    yo2 = tanh(vo2./2); % Network Output
    % Output Layer
    e2 = d(n) - yo2;
    Phiprime2 = 2 .* exp(-vo2) ./ ((1 + exp(-vo2)).^2);
    deta2 = gamma * delta2; % Learning rate adaptation of Output Layer
    delta2 = e2 * Phiprime2;
    deta2 = deta2 * delta2;
    eta2 = eta2 + deta2;
    dW21 = alpha .* dW21 + yh1 .* eta2 .* delta2;
    % Hidden Layer
    Phiprime1 = 2 .* exp(-vh1) ./ ((1 + exp(-vh1)).^2);
    deta1 = gamma * delta1; % Learning rate adaptation of Hidden Layer
  
    delta1 = W21 .* Phiprime1 .* delta2;
    deta1 = deta1 .* delta1;
    eta1 = eta1 + deta1;
    dW10 = alpha * dW10 + eta1 .* delta1 * Xtrain(n,:);
    W21 = W21 + dW21; % Adapting Weights in Output Layer
    W10 = W10 + dW10; % Adapting Weights in Hidden Layer
  end
end
% Generating Test Samples
Nrep = 100;
for x = 1:Nrep
  Data1 = generateData([0,0],1,250);
  Data2 = generateData([2,0],2,250);
  Xtest = [Data1;Data2];
  TestLabel(1:250,1) = 1;
  TestLabel(251:500,1) = 2;
  % Calculating Output
  OutputClass = zeros(size(Xtest,1),1);
  for n = 1:size(Xtest,1) % Test Data loop
    vh1 = W10 * Xtest(n,:)';
    yh1 = tanh(vh1./2); % Hidden Layer Output
    vo2 = W21' * yh1;
    yo2 = tanh(vo2./2); % Network Output
    if yo2 > 0
      OutputClass(n,1) = 1;
    else
      OutputClass(n,1) = 2;
    end
  end
  Result = (TestLabel == OutputClass);
  Eval(x) = size(find(Result),1) * 100 / max(size(Result));
end
disp(['Classification error is: ',num2str(mean(Eval)),'.']);