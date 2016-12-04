clc;clear all;close all;
No = 1000; % number of Training Samples
U = randn(No,1); % Random Inputs
Yp = zeros(No,1); % Output Vector Initialization
Xtrain = zeros(No,5); % Input Vector initialization
d = zeros(No,1); % Desired Value
for k = 3:No-1
Yp(k+1) = f1(Yp(k),Yp(k-1),Yp(k-2),U(k),U(k-1));
d(k+1) = Yp(k+1);
Xtrain(k+1,:) = [Yp(k),Yp(k-1),Yp(k-2),U(k),U(k-1)];
end
%trai




n=5;% input number of input Nodes
w=0.2*randn(n,1);
err=zeros(1,No);
for k=1:300
r=randperm(1000);
Xtrain= Xtrain(r,:);
d=d(r);
for i=1:No
etha=0.01;% learning rate is constatnt
v1=w'*Xtrain(i,:)';
o1=tanhh(v1);
err(i)=d(i)-o1;
w=w+(etha)*(err(i)*Xtrain(i,:)');
end
learn_errorate(k)=sum(err.^2)/No;
end
plot(learn_errorate);
xlabel('epoch');
ylabel('mean square error')
%% test
for k = 1:No/2
u(k) = sin(2*k*pi/250);
end
for k = No/2+1:No
u(k) = 0.8*sin(2*k*pi/250) + 0.2*sin(2*k*pi/25);





end
yo = zeros(size(u));
U = zeros(No,n);

for k = 3:No-1 % System output calculation
Yp(k+1) = f1(Yp(k),Yp(k-1),Yp(k-2),u(k),u(k-1));
U(k+1,:) = [Yp(k),Yp(k-1),Yp(k-2),u(k),u(k-1)];
end
Yo=zeros(No,1);
Uu= [zeros(No,1),zeros(No,1),zeros(No,1),U(:,4),U(:,5)];
o1=0;
for j=3:No-1
v1=w'*Uu(j,:)';
o1=tanhh(v1);
Uu(j,1)=o1;
Yo(j,1)=o1;
end
figure
plot(1:No,Yo,'r');
hold on
plot(Yp); hold off