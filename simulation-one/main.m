clear ; close all; clc;
pkg load statistics;

class1 = generateData(mean=[0 0], variance=1, N=250);
class2 = generateData(mean=[2 0], variance=4, N=250);

%plotData(class1, class2);

Xtrain = [class1; class2];
d(1:250, 1) = 1;
d(251:500, 1) = 2;

Nh = 10; % Number of hidden layer neurons
Epoch = 100; % Number of Epochs
%eta = 0.1;
alpha = [0, 0.1, 0.2, 0.4 ,0.6, 0.8];
eta = [0.1, 0.1, 0.3 ,0.5, 0.8, 1.2];

W10(:, 1) = 2 * rand(Nh, 1) - 1;
W10(:, 2) = 2 * rand(Nh, 1) - 1;
W21 = 2 * rand(Nh, 1) - 1;
dW21 = 0;
dW10 = 0;


for param = 1:6 % Changing Parameter
    for ep = 1:Epoch % Epochs loop
        for n = 1:size(Xtrain, 1) % Training Data loop
            vh1 = W10 * Xtrain(n,:)';
            yh1 = tanh(vh1./2); % Hidden Layer Output
            vo2 = W21' * yh1;
            yo2 = tanh(vo2./2); % Network Output
            % Output
            e2 = d(n) - yo2;
            e(n) = e2;
            Phiprime2 = 2 .* exp(-vo2) ./ ((1 + exp(-vo2)).^2);
            delta2 = e2 * Phiprime2;
            dW21 = alpha(param) .* dW21 + yh1 .* eta(1) .* delta2;
            % Hidden
            Phiprime1 = 2 .* exp(-vh1) ./ ((1 + exp(-vh1)).^2);
            delta1 = W21 .* Phiprime1 .* delta2;
            dW10 = alpha(param) * dW10 + eta(1) * delta1 * Xtrain(n,:);
            W21 = W21 + dW21;
            W10 = W10 + dW10; 
        end
        % ep
        % param
        MSE(ep, param) = sum(e .^ 2) / numel(e);
        % mean(e.^2);
    end
end

figure;
plot(MSE(:,1),'b');xlabel('Epoch');
ylabel('MSE');title('Mean Square Error');
hold on
plot(MSE(:,2),'r');
plot(MSE(:,3),'g');
plot(MSE(:,4),'y');
plot(MSE(:,5),'c');
plot(MSE(:,6),'m');

%legend('\eta = 0','\eta = 0.4','\eta = 0.7','\eta = 0.9');
% 0.05, 0.1, 0.3 ,0.5, 0.8, 1.2
% alpha = [0, 0.1, 0.2, 0.4 ,0.6, 0.8];

legend('\alpha = 0','\alpha = 0.1','\alpha = 0.2','\alpha = 0.4', '\alpha = 0.6', '\alpha = 0.8');

% Generating Test Samples
Nrep = 100;
for x = 1:Nrep
  Data1 = generateData(mean=[0,0], variance=1, N=250);
  Data2 = generateData(mean=[2,0], variance=2, N=250);

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
disp(['Classification error is: ',num2str(100 - sum(Eval) / numel(Eval)),'.']);