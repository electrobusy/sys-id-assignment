% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: FF_ex_4_4.m

clear all 
close all

seed = 1234;
rng(seed);

%% Load the KF data:

load('reconstructed_data_KF.mat');

% % Data choice: 
% -- Extended Kalman Filter:
% Z_k_rec = Z_k_rec_ext;
% data_set = 'EKF';
%  -- Iterative Kalman Filter:
Z_k_rec = Z_k_rec_it;
data_set = 'IEKF';

%% Load functions from parameter estimator (Use the OLS algorithm):

addpath(genpath('../2_Par_Est'));

%% Polynomial order using the multinomial theorem:
% % NOTE: This polynomial will yield a certain number of coeficients that
% is not decided by the used, its is an "outcome" of the theorem:
poly_order = 6;

n_coef = sum(0:poly_order+1); % Using a property of the Pascal triangle. 
                              % Because the Newton Binomial has d+1 terms (think in terms 
                              % of the Pascal Triangle, taking into account the number of 
                              % elements in a row of the triangle). So the idea
                              % is to sum all the elements of the Pascal
                              % triangle.

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 2;     % Fixed -> 2 inputs
HN = n_coef;   % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^2; % Learning rate (initial one)
N_epochs = 400; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

% -- Approximation accuracy:
treshold = 0.09;

%% Data management (later for the LM algorithm):

% % Training/Valitation/Testing data:
train_pct = 0.8; % percentage of training data to be used
val_pct = 0.1; % percentage of validation data to be used

train_size = floor(size(Z_k_rec,1)*train_pct); % number of training data samples
val_size = floor(size(Z_k_rec,1)*val_pct); % number of validation data samples

% - Split dataset in Training and Testing data
% % -- METHOD 1: Use the randperm function
% % Shuffling the data at each epoch:
% index_shuff = randperm(length(Cm)); % index for shuffling
% Cm_shuff = Cm(index_shuff); % Cm is shuffled
% Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled
% % Separate the training and testing data:
% % -- Cm
% Cm_train = Cm_shuff(1:train_size)'; % Training target
% Cm_val = Cm_shuff(train_size+1:train_size+val_size); % Validation target
% Cm_test = Cm_shuff(train_size+val_size+1:end)'; % Testing target
% % -- input
% input_train = Z_k_rec_shuff(1:train_size,[1 2])'; % Training input
% input_val = Z_k_rec_shuff(train_size+1:train_size+val_size,[1 2])'; % Training input
% input_test = Z_k_rec_shuff(train_size+val_size+1:end,[1 2])'; % Testing input


% % - Split dataset in Training and Testing data
% -- METHOD 2: Use the dividerand function
trainRatio = train_pct;
valRatio = val_pct;
testRatio = 1 - trainRatio - valRatio;
[trainInd,valInd,testInd] = dividerand(length(Cm),trainRatio,valRatio,testRatio);
% Training data:
input_train = Z_k_rec(trainInd,[1 2])';
Cm_train = Cm(trainInd)';
% Validation data:
input_val = Z_k_rec(valInd,[1 2])';
Cm_val = Cm(valInd)';
% Testing data:
input_test = Z_k_rec(testInd,[1 2])';
Cm_test = Cm(testInd)';

%% Ex_4_4 - Approximation accuracy

% % -  Transpose the data input and give it to another variable (more readable and for plotting):
net_input = Z_k_rec(:,[1 2])';

% NOTE: accuracy here mean training accuracy, but you can change it to
% testing ou validation too, by just taking into account the following:
% -- training - index = 1
% -- validation - index = 2
% -- testing - index = 3
% Change the index number for the accuracy you would like to check upon
% learning.
index = 1;

%% 1 - Compare approximation accuracy of RBF and FF and required learning iterations to achieve certain level of accuracy: 

% % % ---- RBF NETWORK ----

% -- Create the net
net_RBF = createNet('rbf',IN,HN,ON,mu,N_epochs);

% -- Learning rate increase or decrease: 
net_RBF.trainParam.mu_dec = 1/alpha;
net_RBF.trainParam.mu_inc = alpha;

% -- Cost variable initialization: 
cost_LM_RBF = zeros(N_epochs+1,3); % [cost_train cost_val cost_test]

% -- Use adaptive learning learning algorithm:
% - Forward prop:
output_train_RBF = simNet(net_RBF,input_train,net_RBF.name);
output_val_RBF = simNet(net_RBF,input_val,net_RBF.name);
output_test_RBF = simNet(net_RBF,input_test,net_RBF.name);
% - Compute current cost:
cost_LM_RBF(1,1) = (1/2)*sum((Cm_train - output_train_RBF.Y2).^2); % training cost
cost_LM_RBF(1,2) = (1/2)*sum((Cm_val - output_val_RBF.Y2).^2); % validation cost
cost_LM_RBF(1,3) = (1/2)*sum((Cm_test - output_test_RBF.Y2).^2); % testing cost

% -- Auxiliary network:
net_aux_RBF = net_RBF;

% -- Initialize the counter for the total number of learning iterations to achieve certain accuracy: 
it_RBF = 0;

fprintf('LEVENBERG-MADQUARDT - RBF\n');
fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_RBF(1,1), cost_LM_RBF(1,2), cost_LM_RBF(1,3), net_aux_RBF.trainParam.mu);
        
for i = 1:N_epochs
    % - Forward prop:
    output_train_RBF = simNet(net_aux_RBF,input_train,net_RBF.name);

   % - Backpropagation and network update:
    lambda = net_aux_RBF.trainParam.mu;
    [h_LM,grad] = LMDerMat(net_aux_RBF,output_train_RBF,input_train,Cm_train,lambda);
    
    % - Update the network once to determine the errors: 
    net_aux_RBF = LM_update(net_aux_RBF,h_LM);
    
    % - Compute training output:
    output_train_RBF = simNet(net_aux_RBF,input_train,net_aux_RBF.name);
    % - Compute validation output:
    output_val_RBF = simNet(net_aux_RBF,input_val,net_aux_RBF.name);
    % - Compute testing output:
    output_test_RBF = simNet(net_aux_RBF,input_test,net_aux_RBF.name);

    % - Compute training cost:
    cost_LM_RBF(i+1,1) = (1/2)*sum((Cm_train - output_train_RBF.Y2).^2); 
    % - Compute validation cost:
    cost_LM_RBF(i+1,2) = (1/2)*sum((Cm_val - output_val_RBF.Y2).^2);
    % - Compute testing cost:
    cost_LM_RBF(i+1,3) = (1/2)*sum((Cm_test - output_test_RBF.Y2).^2);
    
    % % -----------------------------------------------------------------------------   
    % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
    if cost_LM_RBF(i+1,1) < cost_LM_RBF(i,1)
        % Keep the changes: 
        net_RBF = net_aux_RBF; 
        % Increase learning rate:
        net_aux_RBF.trainParam.mu = net_aux_RBF.trainParam.mu_dec*net_aux_RBF.trainParam.mu;
    elseif cost_LM_RBF(i+1,1) >= cost_LM_RBF(i,1) 
        % Save current mu to update:
        mu_aux = net_aux_RBF.trainParam.mu;
        % Reject the changes:
        net_aux_RBF = net_RBF; % This net contains mu from the previous if
        % Decrease learning rate:
        net_aux_RBF.trainParam.mu = net_aux_RBF.trainParam.mu_inc*mu_aux;
        % Keep the previous cost function:
        cost_LM_RBF(i+1,:) = cost_LM_RBF(i,:);
    end
    if cost_LM_RBF(i+1,index) < treshold && it_RBF == 0
        it_RBF = i;
    end
    if cost_LM_RBF(i+1,1) < net_aux_RBF.trainParam.goal || sum(abs(grad)) < net_aux_RBF.trainParam.min_grad || net_aux_RBF.trainParam.mu > net_RBF.trainParam.mu_max
        break;
    end
    fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM_RBF(i+1,1), cost_LM_RBF(i+1,2), cost_LM_RBF(i+1,3), net_aux_RBF.trainParam.mu);
end

% % % ---- FF NETWORK ----

% -- Create the net
net_FF = createNet('feedforward',IN,HN,ON,mu,N_epochs);

% -- Learning rate increase or decrease: 
net_FF.trainParam.mu_dec = 1/alpha;
net_FF.trainParam.mu_inc = alpha;

% -- Cost variable initialization: 
cost_LM_FF = zeros(N_epochs+1,3); % [cost_train cost_val cost_test]

% % -- Use adaptive learning learning algorithm:
% - Forward prop:
output_train_FF = simNet(net_FF,input_train,net_FF.name);
output_val_FF = simNet(net_FF,input_val,net_FF.name);
output_test_FF = simNet(net_FF,input_test,net_FF.name);
% - Compute current cost:
cost_LM_FF(1,1) = (1/2)*sum((Cm_train - output_train_FF.Y2).^2); % training cost
cost_LM_FF(1,2) = (1/2)*sum((Cm_val - output_val_FF.Y2).^2); % validation cost
cost_LM_FF(1,3) = (1/2)*sum((Cm_test - output_test_FF.Y2).^2); % testing cost

% -- Auxiliary network:
net_aux_FF = net_FF;

% -- Initialize the counter for the total number of learning iterations to achieve certain accuracy: 
it_FF = 0;

fprintf('LEVENBERG-MADQUARDT - FF\n');
fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_FF(1,1), cost_LM_FF(1,2), cost_LM_FF(1,3), net_aux_FF.trainParam.mu);

for j = 1:N_epochs 
    % - Forward prop:
    output_train_FF = simNet(net_aux_FF,input_train,net_FF.name);

    % - Backpropagation and network update:
    lambda = net_aux_FF.trainParam.mu;
    [h_LM,grad] = LMDerMat(net_aux_FF,output_train_FF,input_train,Cm_train,lambda);
    
    % - Update the network once to determine the errors: 
    net_aux_FF = LM_update(net_aux_FF,h_LM);
    
    % - Compute training output:
    output_train_FF = simNet(net_aux_FF,input_train,net_aux_FF.name);
    % - Compute validation output:
    output_val_FF = simNet(net_aux_FF,input_val,net_aux_FF.name);
    % - Compute testing output:
    output_test_FF = simNet(net_aux_FF,input_test,net_aux_FF.name);

    % - Compute training cost:
    cost_LM_FF(j+1,1) = (1/2)*sum((Cm_train - output_train_FF.Y2).^2); 
    % - Compute validation cost:
    cost_LM_FF(j+1,2) = (1/2)*sum((Cm_val - output_val_FF.Y2).^2);
    % - Compute testing cost:
    cost_LM_FF(j+1,3) = (1/2)*sum((Cm_test - output_test_FF.Y2).^2);
    
    % % -----------------------------------------------------------------------------   
    % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
    if cost_LM_FF(j+1,1) < cost_LM_FF(j,1)
        % Keep the changes: 
        net_FF = net_aux_FF; 
        % Increase learning rate:
        net_aux_FF.trainParam.mu = net_aux_FF.trainParam.mu_dec*net_aux_FF.trainParam.mu;
    elseif cost_LM_FF(j+1,1) >= cost_LM_FF(j,1) 
        % Save current mu to update:
        mu_aux = net_aux_FF.trainParam.mu;
        % Reject the changes:
        net_aux_FF = net_FF; % This net contains mu from the previous if
        % Decrease learning rate:
        net_aux_FF.trainParam.mu = net_aux_FF.trainParam.mu_inc*mu_aux;
        % Keep the previous cost function:
        cost_LM_FF(j+1,:) = cost_LM_FF(j,:);
    end
    if cost_LM_FF(j+1,index) < treshold && it_FF == 0
        it_FF = j;
    end
    if cost_LM_FF(j+1,1) < net_aux_FF.trainParam.goal || sum(abs(grad)) < net_aux_FF.trainParam.min_grad || net_aux_FF.trainParam.mu > net_FF.trainParam.mu_max
        break;
    end
    fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / Learning rate = %f \n', j, cost_LM_FF(j+1,1), cost_LM_FF(j+1,2), cost_LM_FF(j+1,3), net_aux_FF.trainParam.mu);
end

% - Plot of the costs: 
num_fig = 1;
figure(num_fig);
semilogy(1:i,cost_LM_RBF(2:(i+1),1),1:i,cost_LM_RBF(2:(i+1),2),1:i,cost_LM_RBF(2:(i+1),3));
hold on; 
semilogy(1:j,cost_LM_FF(2:(j+1),1),1:j,cost_LM_FF(2:(j+1),2),1:j,cost_LM_FF(2:(j+1),3));
xL = get(gca,'XLim'); % Treshold -- RBF and FF
p3= line(xL,[treshold treshold],'Linestyle','--','Color','k',...
    'Linewidth',1);
yL = get(gca,'YLim'); % Required number of iterations -- RBF and FF
p1= line([it_RBF it_RBF],yL,'Linestyle','-','Color','r',...
    'Linewidth',1);
p2= line([it_FF it_FF],yL,'Linestyle','-.','Color','b',...
    'Linewidth',1);
legend('Training error - RBF','Validation error - RBF','Testing error - RBF', ... 
    'Training error - FF','Validation error - FF','Testing error - FF', ...
    ['Treshold = ' num2str(treshold)], ['# of iterations (RBF) = ' num2str(it_RBF)], ...
    ['# of iterations (FF) = ' num2str(it_FF)]);
xlabel('Number of epochs [-]');
ylabel('Cost [-]');
title(['Method LM - Batch training (batch of ' num2str(train_pct*100)  '% of entire dataset)']);

%% 2 - Compare results with the polynomial model identified in part 2:

% %  -- Define the regression matrix A with the reconstructed data:
A = reg_matrix(net_input(1,:)',net_input(2,:)',poly_order);

% % -- Apply Ordinary Least-Squares Estimator (OLS) (or other):
[est_par_OLS,output_OLS] = ord_least_squares(A,Cm);

% % -- Compute total cost:
cost_total_OLS = (1/2)*sum((Cm - output_OLS).^2);

fprintf('\n OLS Algorithm performance: \n');
fprintf('OLS cost after identifying entire data set: %d \n', cost_total_OLS);

% ---- USE ONLY THE SAME AMOUNT OF TRAINING DATA OF THE NN TO IDENTIFY THE MODEL:
% %  -- Define the regression matrix A with the reconstructed data:
A_train = reg_matrix(input_train(1,:),input_train(2,:),poly_order);

% % -- Apply Ordinary Least-Squares Estimator (OLS) (or other):
[est_par_OLS,output_train_OLS] = ord_least_squares(A_train,Cm_train');

% % -- With the estimated parameters, determine the output for validation and testing data:
% - Define regression matrix:
A_val = reg_matrix(input_val(1,:),input_val(2,:),poly_order);
A_test = reg_matrix(input_test(1,:),input_test(2,:),poly_order);
% - Apply polynomial model:
output_val_OLS = A_val * est_par_OLS;
output_test_OLS = A_test * est_par_OLS;

% % -- Now determine the cost: 
% - Compute training cost:
cost_OLS(1) = (1/2)*sum((Cm_train' - output_train_OLS).^2);
% - Compute validation cost:
cost_OLS(2) = (1/2)*sum((Cm_val' - output_val_OLS).^2);
% - Compute testing cost:
cost_OLS(3) = (1/2)*sum((Cm_test' - output_test_OLS).^2);

fprintf('OLS cost after identifying %d percent of the data set: \n', train_pct*100);
fprintf('Training cost: %d / Validation Cost = %d / Testing Cost = %d \n', cost_OLS(1), cost_OLS(2), cost_OLS(3));

