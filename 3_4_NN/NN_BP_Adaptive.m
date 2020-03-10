% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: NN_BP_Adaptive.m

clear all 
close all

%% DATA: -- Simulated data

dt = 0.05;
x = (0:dt:2*pi)';
func = x.^2; % -- choose the function you would like to approximate
y = func + 0.1*rands(length(x),1);

%% Define the seed (using random functions)
seed = 123456;
rng(seed);

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 1;     % Fixed -> one inputs
HN = 20;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^-2; % Learning rate (initial one)
N_epochs = 10000; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

%% Data management (later for the BP algorithm):

% % - Split dataset in Training and Testing data
% -- METHOD: Use the dividerand function
trainRatio = 0.7;
valRatio = 0.1;
testRatio = 1 - trainRatio - valRatio;
[trainInd,valInd,testInd] = dividerand(length(y),trainRatio,valRatio,testRatio);

% Training data:
input_train = x(trainInd)';
y_train = y(trainInd)';
% Validation data:
input_val = x(valInd)';
y_val = y(valInd)';
% Testing data:
input_test = x(testInd)';
y_test = y(testInd)';

%% BACK-PROP ALGORITHM:
% % -----------------------------------------------------------------------
% % -- BATCH TRAINING ALGORITHM (ENTIRE DATASET) --> ENITRE BATCH GRADIENT DESCENT
% % -----------------------------------------------------------------------
  
% -- Create the net
net = createNet('rbf',IN,HN,ON,mu,N_epochs);

% -- Learning rate increase or decrease: 
net.trainParam.mu_dec = 1/alpha;
net.trainParam.mu_inc = alpha;

% -- Cost variable initialization: 
cost_BP = zeros(N_epochs+1,3); % [cost_train cost_val cost_test]

% % -- Use adaptive learning learning algorithm:
% - Forward prop:
output_train = simNet(net,input_train,net.name);
output_val = simNet(net,input_val,net.name);
output_test = simNet(net,input_test,net.name);

% - Compute current training/validation/testing cost:
cost_BP(1,1) = (1/2)*sum((y_train - output_train.Y2).^2); % training cost
cost_BP(1,2) = (1/2)*sum((y_val - output_val.Y2).^2); % validation cost
cost_BP(1,3) = (1/2)*sum((y_test - output_test.Y2).^2); % testing cost

% -- Auxiliary network:
net_aux = net;

fprintf('BACKPROPAGATION\n');
fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_BP(1,1), cost_BP(1,2), cost_BP(1,3), net_aux.trainParam.mu);

for i = 1:N_epochs 
    % - Forward prop:
    output_train = simNet(net_aux,input_train,net.name);
    
    % - Backpropagation and network update:
    [update,grad] = BackPropDerMat(net_aux,output_train,input_train,y_train,net_aux.trainParam.mu);
    
    % - Update the network once to determine the errors: 
    net_aux = BP_update(net_aux,update);
    
    % - Compute training output:
    output_train = simNet(net_aux,input_train,net_aux.name);
    % - Compute validation output:
    output_val = simNet(net_aux,input_val,net_aux.name);
    % - Compute testing output:
    output_test = simNet(net_aux,input_test,net_aux.name);

    % - Compute training cost:
    cost_BP(i+1,1) = (1/2)*sum((y_train - output_train.Y2).^2); 
    % - Compute validation cost:
    cost_BP(i+1,2) = (1/2)*sum((y_val - output_val.Y2).^2);
    % - Compute testing cost:
    cost_BP(i+1,3) = (1/2)*sum((y_test - output_test.Y2).^2);
    
    % % -----------------------------------------------------------------------------   
    % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
    if cost_BP(i+1,1) < cost_BP(i,1) 
        % Keep the changes: 
        net = net_aux; 
        % Increase learning rate:
        net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
    elseif cost_BP(i+1,1) >= cost_BP(i,1) % || isnan(cost_BP(i+1,1)) == 1
        % Save current mu to update:
        mu_aux = net_aux.trainParam.mu;
        % Reject the changes:
        net_aux = net; % This net contains mu from the previous if
        % Decrease learning rate:
        net_aux.trainParam.mu = net_aux.trainParam.mu_dec*mu_aux;
        % Keep the previous cost function:
        cost_BP(i+1,:) = cost_BP(i,:);
    end
    if cost_BP(i+1,1) < net_aux.trainParam.goal || sum(abs(grad)) < net_aux.trainParam.min_grad || net_aux.trainParam.mu > net.trainParam.mu_max
        % cost_BP(i+1,:) = cost_BP(i,:);
        break;
    end
    fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_BP(i+1,1), cost_BP(i+1,2), cost_BP(i+1,3), net_aux.trainParam.mu);
end

% - Plots: 
num_fig = 1;
figure(num_fig);
semilogy(1:i,cost_BP(2:(i+1),1),1:i,cost_BP(2:(i+1),2),1:i,cost_BP(2:(i+1),3));
legend('Training error','Validation error','Testing error');
xlabel('Number of epochs [-]');
ylabel('Cost [-]');
title('Method BP - Batch training (batch of the entire dataset)');

num_fig = num_fig + 1;
figure(num_fig);
% - Compute training output:
output_train = simNet(net,input_train,net_aux.name);
% - Compute validation output:
output_val = simNet(net,input_val,net_aux.name);
% - Compute testing output:
output_test = simNet(net,input_test,net_aux.name);

plot(x,y);
hold on;
plot(input_train,output_train.Y2,'r+');
hold on;
plot(input_val,output_val.Y2,'b+');
hold on;
plot(input_test,output_test.Y2,'g+');
if strcmp(net_aux.name,'rbf') == 1
    hold on;
    scatter(net.centers,zeros(size(net.centers,1),1),'mo');
    if valRatio ~= 0
        legend('Target','Train','Validation','Test','RBF - Centers');
    else
        legend('Target','Train','Test','RBF - Centers');
    end
else 
    if valRatio ~= 0
        legend('Target','Train','Validation','Test');
    else
        legend('Target','Train','Test');
    end
end
xlabel('x');
ylabel('y');