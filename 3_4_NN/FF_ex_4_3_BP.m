% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: FF_ex_4_3_BP.m

clear all 

%% Load the KF data:

load('reconstructed_data_KF.mat');

% % Data choice: 
% -- Extended Kalman Filter:
% Z_k_rec = Z_k_rec_ext;
% data_set = 'EKF';
%  -- Iterative Kalman Filter:
Z_k_rec = Z_k_rec_it;
data_set = 'IEKF';

%% Define the seed (using random functions)
% seed = 123456;
% rng(seed);

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 2;     % Fixed -> 2 inputs
HN = 40;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^-1; % Learning rate (initial one)
N_epochs = 1000; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

% Number of weight initializations
num_init = 5;

% % Max number of neurons to test - Sensibility and optimization part
max_HN = 10;

%% Data management (later for the LM algorithm):

% % Training/Valitation/Testing data:
train_pct = 0.9; % percentage of training data to be used
val_pct = 0.05; % percentage of validation data to be used

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

%% Ex_4_3 - Optimization algorithm of the RBF network in terms of total number of neurons

% -- Net input (Later for the 3-D plot)
net_input = Z_k_rec(:,[1 2])';

% -- Optimal cost variable initialization: 
cost_opt = zeros(max_HN,4); % [cost_train cost_val cost_test cost_all]

% -- Cost after training for each initialization:
cost_init = zeros(num_init,4); % [cost_train cost_val cost_test cost_all]

for n = 1:max_HN
    
    % % -----------------------------------------------------------------------   
    % % -- USED THE BATCH TRAINING ALGORITHM (ENTIRE DATASET) --> BATCH GRADIENT DESCENT
    % % ----------------------------------------------------------------------- 

    % -- Recall the influence of the seed: 
    % rng(seed);
    
    for j = 1:num_init
        
         % -- Create net:
        net = createNet('feedforward',IN,n,ON,mu,N_epochs);
        
        % -- Learning rate increase or decrease (upon training)
        net.trainParam.mu_dec = 1/alpha;
        net.trainParam.mu_inc = alpha;
        
        % -- Cost variable initialization: 
        cost_LM = zeros(N_epochs+1,4); % [cost_train cost_val cost_test cost_all]

        % % -- Use adaptive learning learning algorithm:
        % - Forward prop:
        output_train = simNet(net,input_train,net.name);
        output_val = simNet(net,input_val,net.name);
        output_test = simNet(net,input_test,net.name);
        output_all = simNet(net,net_input,net.name);

        % - Compute current training/validation/testing cost:
        cost_LM(1,1) = 1/(2)*sum((Cm_train - output_train.Y2).^2); % training cost
        cost_LM(1,2) = 1/(2)*sum((Cm_val - output_val.Y2).^2); % testing cost
        cost_LM(1,3) = 1/(2)*sum((Cm_test - output_test.Y2).^2); % testing cost
        cost_LM(1,4) = 1/(2)*sum((Cm' - output_all.Y2).^2); % all data cost

        % -- Auxiliary network:
        net_aux = net;
        
        % fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f / Learning rate = %f \n', 0, cost_LM(1,1), cost_LM(1,2), cost_LM(1,3), cost_LM(1,4), net_aux.trainParam.mu);
        
        for i = 1:N_epochs
            % - Forward prop:
            output_train = simNet(net_aux,input_train,net.name);
            
            % - Backpropagation and network update:
            lambda = net_aux.trainParam.mu;
            % - Backpropagation and network update:
            [update,grad] = BackPropDerMat(net_aux,output_train,input_train,Cm_train,net_aux.trainParam.mu);
    
            % - Update the network once to determine the errors: 
            net_aux = BP_update(net_aux,update);
            
            % - Compute training output:
            output_train = simNet(net_aux,input_train,net_aux.name);
            % - Compute validation output:
            output_val = simNet(net_aux,input_val,net_aux.name);
            % - Compute testing output:
            output_test = simNet(net_aux,input_test,net_aux.name);
            % - Computer all data output:
            output_all = simNet(net_aux,net_input,net.name);
            
            % - Compute training cost:
            cost_LM(i+1,1) = 1/(2)*sum((Cm_train - output_train.Y2).^2);
            % - Compute training cost:
            cost_LM(i+1,2) = 1/(2)*sum((Cm_val - output_val.Y2).^2);
            % - Compute testing cost:
            cost_LM(i+1,3) = 1/(2)*sum((Cm_test - output_test.Y2).^2);
            % - Compute cost of all data:
            cost_LM(i+1,4) = 1/(2)*sum((Cm' - output_all.Y2).^2); 
            
            % % -----------------------------------------------------------------------------
            % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH
            if cost_LM(i+1,1) < cost_LM(i,1)
                % Keep the changes:
                net = net_aux;
                % Increase learning rate:
                net_aux.trainParam.mu = net_aux.trainParam.mu_dec*net_aux.trainParam.mu;
            elseif cost_LM(i+1,1) >= cost_LM(i,1)
                % Save current mu to update:
                mu_aux = net_aux.trainParam.mu;
                % Reject the changes:
                net_aux = net; % This net contains mu from the previous if
                % Decrease learning rate:
                net_aux.trainParam.mu = net_aux.trainParam.mu_inc*mu_aux;
                % Keep the previous cost function:
                cost_LM(i+1,:) = cost_LM(i,:);
            end
            if cost_LM(i+1,1) < net_aux.trainParam.goal || sum(abs(grad)) < net_aux.trainParam.min_grad || net_aux.trainParam.mu > net.trainParam.mu_max
                break;
            end
            % fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f / Learning rate = %f \n', i, cost_LM(i+1,1), cost_LM(i+1,2), cost_LM(i+1,3), cost_LM(i+1,4), net_aux.trainParam.mu);
        end
        cost_init(j,1) = cost_LM(i+1,1);
        cost_init(j,2) = cost_LM(i+1,2);
        cost_init(j,3) = cost_LM(i+1,3);
        cost_init(j,4) = cost_LM(i+1,4);
    end
%     clf;
%     num_fig = 1;
%     plot_FF_inputs(Cm,num_fig,['FF-NN - ' data_set ' -> For ' num2str(n) ' neurons'],Z_k_rec(:,[1 2]),input_train',input_val',input_test',Cm_train',Cm_val',Cm_test');
    
    cost_opt(n,1) = mean(cost_init(:,1));
    cost_opt(n,2) = mean(cost_init(:,2));
    cost_opt(n,3) = mean(cost_init(:,3));
    cost_opt(n,4) = mean(cost_init(:,4));
    fprintf('# of Neurons = %d : Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f \n', n, cost_opt(n,1), cost_opt(n,2), cost_opt(n,3), cost_opt(n,4));
end

% % Plots:
num_fig = 1;
figure(num_fig);
set(gca, 'YScale', 'log');
semilogy(1:max_HN,cost_opt(:,1),1:max_HN,cost_opt(:,2),1:max_HN,cost_opt(:,3),1:max_HN,cost_opt(:,4));
[~,idx_HN] = min(cost_opt(:,3));
yL = get(gca,'YLim'); % Required number of iterations -- RBF and FF
p2= line([idx_HN idx_HN],yL,'Linestyle','-.','Color','b',...
    'Linewidth',1);
legend('Training','Validation','Testing','All data',['Optimal HN = ' num2str(idx_HN)]);
xlabel('Number of neurons [-]');
ylabel('Cost after training [-]');
title('Optimization of number of neurons');