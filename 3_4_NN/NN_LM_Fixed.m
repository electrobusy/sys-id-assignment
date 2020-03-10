% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: NN_LM_Fixed.m

clear all 
close all

%% DATA: -- Simulated data

dt = 0.1;
x = (0:dt:2*pi)';
func = x.^2; % -- choose the function you would like to approximate
y = func + 0.1*rands(length(x),1);

%% Define the seed (using random functions)
% seed = 12345678;
% rng(seed);

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 1;     % Fixed -> one inputs
HN = 20;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^2; % Learning rate (initial one)
N_epochs = 10000; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate [NO NEED IN THIS CASE]

%% Data management (later for the LM algorithm):

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

%% LEVENBERG-MADQUARDT ALGORITHM:

% -- Create the net
net = createNet('rbf',IN,HN,ON,mu,N_epochs);

% -- Learning rate increase or decrease: 
net.trainParam.mu_dec = 1/alpha;
net.trainParam.mu_inc = alpha;

% -- Cost variable initialization: 
cost_LM = zeros(N_epochs+1,3); % [cost_train cost_val cost_test]

% % -- Use adaptive learning learning algorithm:
% - Forward prop:
output_train = simNet(net,input_train,net.name);
output_val = simNet(net,input_val,net.name);
output_test = simNet(net,input_test,net.name);

% - Compute current training/validation/testing cost:
cost_LM(1,1) = (1/(2*length(y_train)))*sum((y_train - output_train.Y2).^2); % training cost
cost_LM(1,2) = (1/(2*length(y_val)))*sum((y_val - output_val.Y2).^2); % validation cost
cost_LM(1,3) = (1/(2*length(y_test)))*sum((y_test - output_test.Y2).^2); % testing cost

% -- Auxiliary network:
net_aux = net;

fprintf('LEVENBERG MADQUARDT\n');
fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM(1,1), cost_LM(1,2), cost_LM(1,3), net_aux.trainParam.mu);

for i = 1:N_epochs 
    % - Backpropagation using the LM algorithm:
    lambda = net_aux.trainParam.mu;
    [h_LM,grad] = LMDerMat(net_aux,output_train,input_train,y_train,lambda);
    
    % - Update the network once to determine the errors: 
    net_aux = LM_update(net_aux,h_LM);
    
    % - Compute training output:
    output_train = simNet(net_aux,input_train,net_aux.name);
    % - Compute validation output:
    output_val = simNet(net_aux,input_val,net_aux.name);
    % - Compute testing output:
    output_test = simNet(net_aux,input_test,net_aux.name);

    % - Compute training cost:
    cost_LM(i+1,1) = (1/(2*length(y_train)))*sum((y_train - output_train.Y2).^2); 
    % - Compute training cost:
    cost_LM(i+1,2) = (1/(2*length(y_val)))*sum((y_val - output_val.Y2).^2);
    % - Compute testing cost:
    cost_LM(i+1,3) = (1/(2*length(y_test)))*sum((y_test - output_test.Y2).^2);
    
    % % -----------------------------------------------------------------------------   
    % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
    if cost_LM(i+1,1) <= cost_LM(i,1)
        % Save the changes: 
        net = net_aux; 
    % elseif cost_LM(i+1,1) > cost_LM(i,1) 
        % Keep the previous cost function:
    %    cost_LM(i+1,:) = cost_LM(i,:);
    end
    if cost_LM(i+1,1) < net_aux.trainParam.goal || abs(min(grad)) < net_aux.trainParam.min_grad 
        % cost_LM(i+1,:) = cost_LM(i,:);
        break;
    end
    fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM(i+1,1), cost_LM(i+1,2), cost_LM(i+1,3), net_aux.trainParam.mu);
end

% - Plots: 
num_fig = 1;
figure(num_fig);
semilogy(1:i,cost_LM(2:(i+1),1),1:i,cost_LM(2:(i+1),2),1:i,cost_LM(2:(i+1),3));
legend('Training error','Validation error','Testing error');
xlabel('Number of epochs [-]');
ylabel('Cost [-]');
title('Method LM - Batch training (batch of the entire dataset)');

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