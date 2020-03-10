% Código de teste para RBF + LM:

clear all
close all

% Load data
load('reconstructed_data_KF.mat');

% Concatenate alpha and beta:
Z_k_rec = Z_k_rec_ext;

% Add the path: 
addpath(genpath('../2_Par_Est'));

% % Seed
% seed = 123456;
% rng(seed);

% % Neural Net Structure parameters:
IN = 2;     % Fixed -> 2 inputs [FIXED]
HN = 1000;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output [FIXED]

mu = 1*10^-4; % Learning rate (initial one)
N_epochs = 10; % Number of Epochs

% % Learning rate increase or decrease: 
alpha = 10;
net.trainParam.mu_dec = 1/alpha;
net.trainParam.mu_inc = alpha;

% % Neural Net: 
% 1 - Create the net:
net = createNet('rbf',IN,HN,ON,mu,N_epochs);

% 2 - Initialize the cost variable
cost_LM = zeros(N_epochs,2); % [cost_train cost_test]

% 3 - Split dataset in Training and Testing data:
trainRatio = 0.8;
valRatio = 0;
testRatio = 1 - trainRatio;
[trainInd,~,testInd] = dividerand(length(Cm),trainRatio,valRatio,testRatio);

inputTrain = Z_k_rec(trainInd,[1 2])';
targetTrain = Cm(trainInd)';

inputTest = Z_k_rec(testInd,[1 2])';
targetTest = Cm(testInd)';

% % 4 - Use adaptive algorithm:
% -- 1rst Epoch
% - Foward prop:
outputTrain = simNet(net,inputTrain,net.name);
outputTest = simNet(net,inputTest,net.name);

% - Compute current cost
cost_LM(1,1) = (1/2)*sum((targetTrain - outputTrain.Y2).^2); % training cost
cost_LM(1,2) = (1/2)*sum((targetTest - outputTest.Y2).^2); % testing cost

net_aux = net;
fprintf('Epoch %d: Training Error = %f / Testing Error = %f / Learning rate = %f \n', 1, cost_LM(1,1), cost_LM(1,2), net_aux.trainParam.mu);

% - Other Epochs
for i = 2:N_epochs
    % Forward prop. for training set:
    outputTrain = simNet(net_aux,inputTrain,net_aux.name);
    
    % LM algorithm for back prop.:
    net_aux = LMDerMat(net_aux,outputTrain,inputTrain,targetTrain,net_aux.trainParam.mu);
    
    % Compute training and testing cost:
    % - training
    outputTrain = simNet(net_aux,inputTrain,net_aux.name);
    cost_LM(i,1) = (1/2)*sum((targetTrain - outputTrain.Y2).^2);
    % - testing
    outputTest = simNet(net_aux,inputTest,net_aux.name);
    cost_LM(i,2) = (1/2)*sum((targetTest - outputTest.Y2).^2);
    
    fprintf('Epoch %d: Training Error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM(i,1), cost_LM(i,2), net_aux.trainParam.mu);
    
    if cost_LM(i,1) <= cost_LM(i-1,1)
       net = net_aux;
       net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
    elseif cost_LM(i,1) > cost_LM(i-1,1)
       net_aux.trainParam.mu = net_aux.trainParam.mu_dec*net_aux.trainParam.mu;
    % elseif 
    end
end

% % Plots:
num_fig = 1;
figure(num_fig);
plot(1:N_epochs,cost_LM(:,1),1:N_epochs,cost_LM(:,2));
legend('Training error','Testing error');
xlabel('Number of epochs');
ylabel('Cost');

num_fig = num_fig + 1;
outputAll = simNet(net,Z_k_rec(:,[1 2])',net.name);
plot_cm(outputAll.Y2,Z_k_rec(:,[1 2]),num_fig,'RBF - Levenberg-Madquardt');
