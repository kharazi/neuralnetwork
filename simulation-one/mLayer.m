%% Generating Training Samples
clc;clear
Data1 = generateData([0,0],1,250);
Data2 = generateData([2,0],2,250);
Xtrain = [Data1;Data2];
d(1:250,1) = 1; % Desired Value for Output
d(251:500,1) = -1;
% Implementing The Network
Nh = 10; % Number of hidden layer neurons
Epoch = 150; % Number of Epochs
eta = 0.1;
alpha = 0;
% initializing Weights in [-1,1]
W10 = 2 * rand(Nh,2) - 1; % Input -> Hidden Layer 1
W21 = 2 * rand(Nh,Nh) - 1; % Hidden Layer 1 -> Hidden Layer 2
W32 = 2 * rand(Nh,1) - 1; % Hidden Layer 2 -> Output Layer
dW32 = zeros(size(W32));
dW21 = zeros(size(W21));
dW10 = zeros(size(W10));
% Training the Network
for ep = 1:Epoch % Epochs loop
  for n = 1:size(Xtrain,1) % Training Data loop

    vh1 = W10 * Xtrain(n,:)';
    yh1 = tanh(vh1./2); % Hidden Layer 1 Output
    vh2 = W21' * yh1;
    yh2 = tanh(vh2./2); % Hidden Layer 2 Output
    vo3 = W32' * yh2;
    yo3 = tanh(vo3/2); % Network Output
    % Output Layer
    e3 = d(n) - yo3;
    Phiprime3 = 2 .* exp(-vo3) ./ ((1 + exp(-vo3)).^2);
    delta3 = e3 * Phiprime3;
    dW32 = alpha .* dW32 + yh2 .* eta .* delta3;
    % Hidden Layer 2
    Phiprime2 = 2 .* exp(-vh2) ./ ((1 + exp(-vh2)).^2);
    delta2 = Phiprime2 * W32' * delta3;
    for i = 1:size(delta2,2)
      dW21(:,i) = alpha .* dW21(:,i) + eta * yh1' * delta2(:,i);
    end
    % Hidden Layer 1
    Phiprime1 = 2 .* exp(-vh1) ./ ((1 + exp(-vh1)).^2);
    delta1 = W21 * delta2 * Phiprime1;
    dW10 = alpha .* dW10 + eta * delta1 * Xtrain(n,:);
    W32 = W32 + dW32; % Adapting Weights in Output Layer
    W21 = W21 + dW21; % Adapting Weights in Hidden Layer 2
    W10 = W10 + dW10; % Adapting Weights in Hidden Layer 1
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