% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: RBF_ex_3_3_cross_val.m

clear all 
close all

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
HN = 10;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^-3; % Learning rate (initial one)
N_epochs = 25; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

% Number of weight initializations
num_init = 2;

% % Max number of neurons to test - Sensibility and optimization part
max_HN = 20;

%% Data management (later for the algorithm):

% % Shuffling the data at each epoch:
index_shuff = randperm(length(Cm)); % index for shuffling
Cm_shuff = Cm(index_shuff); % Cm is shuffled
Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled

% % - Training/Valitation/Testing data:
train_pct = 0.7; % percentage of training data to be used
val_pct = 0.05; % percentage of validation data to be used

% Number of divisions - cross-validation part -> Each fold includes test + validation data:
k_folds = floor(1/(1 - train_pct));
samples_per_fold = floor(length(Cm)*(1 - train_pct)); % Number of data-points per fold

% Percentage of the samples in the fold that are for validation
testval_val_pct = val_pct*1/(1 - train_pct); 

%% Ex_3_3 - Optimization algorithm of the RBF network in terms of total number of neurons

% -- Net input (Later for the 3-D plot)
net_input = Z_k_rec(:,[1 2])';

% -- Optimal cost variable: 
cost_opt = zeros(max_HN,4); % [cost_train cost_val cost_test cost_all]

% -- Cost after training for each initialization:
cost_init = zeros(num_init,4); % [cost_train cost_val cost_test cost_all]

% -- Cost for each fold:
cost_fold = zeros(k_folds,4); % [cost_train cost_val cost_test cost_all]

for n = 1:max_HN
    % % -----------------------------------------------------------------------   
    % % -- USED THE BATCH TRAINING ALGORITHM (ENTIRE DATASET) 
    % % ----------------------------------------------------------------------- 
    
    % -- Recall the influence of the seed: 
    % rng(seed);
    
    for k = 1:k_folds
        idx_fold = samples_per_fold*(k-1)+1:samples_per_fold*k;
        
        % % - Choose indices for training, validation and testing:
        trainInd = [1:samples_per_fold*(k-1) samples_per_fold*k+1:length(Cm)];
        valInd = idx_fold(1:floor(samples_per_fold*testval_val_pct));
        testInd = idx_fold(floor(samples_per_fold*testval_val_pct)+1:end); 
        
        % Training data:
        input_train = Z_k_rec_shuff(trainInd,[1 2])';
        Cm_train = Cm_shuff(trainInd)';
        % Validation data:
        input_val = Z_k_rec_shuff(valInd,[1 2])';
        Cm_val = Cm_shuff(valInd)';
        % Testing data - Our testing data is the fold
        input_test = Z_k_rec_shuff(testInd,[1 2])';
        Cm_test = Cm_shuff(testInd)';
        % All data:
        input_all = Z_k_rec_shuff(:,[1 2])';
        Cm_all = Cm_shuff';
        
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
            output_all = simNet(net,input_all,net.name);

            % - Compute current training/validation/testing cost:
            cost_LM(1,1) = 1/(2)*sum((Cm_train - output_train.Y2).^2); % training cost
            cost_LM(1,2) = 1/(2)*sum((Cm_val - output_val.Y2).^2); % testing cost
            cost_LM(1,3) = 1/(2)*sum((Cm_test - output_test.Y2).^2); % testing cost
            cost_LM(1,4) = 1/(2)*sum((Cm_all - output_all.Y2).^2); % all data cost

            % -- Auxiliary network:
            net_aux = net;
        
            % fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f / Learning rate = %f \n', 0, cost_LM(1,1), cost_LM(1,2), cost_LM(1,3), cost_LM(1,4), net_aux.trainParam.mu);
            for i = 1:N_epochs
                % - Forward prop:
                output_train = simNet(net_aux,input_train,net.name);
                
                % - Backpropagation and network update:
                lambda = net_aux.trainParam.mu;
                [h_LM,grad] = LMDerMat(net_aux,output_train,input_train,Cm_train,lambda);
                
                % - Update the network once to determine the errors:
                net_aux = LM_update(net_aux,h_LM);
                
                % - Compute training output:
                output_train = simNet(net_aux,input_train,net_aux.name);
                % - Compute validation output:
                output_val = simNet(net_aux,input_val,net_aux.name);
                % - Compute testing output:
                output_test = simNet(net_aux,input_test,net_aux.name);
                % - Computer all data output:
                output_all = simNet(net_aux,input_all,net.name);
                
                % - Compute training cost:
                cost_LM(i+1,1) = 1/(2)*sum((Cm_train - output_train.Y2).^2);
                % - Compute training cost:
                cost_LM(i+1,2) = 1/(2)*sum((Cm_val - output_val.Y2).^2);
                % - Compute testing cost:
                cost_LM(i+1,3) = 1/(2)*sum((Cm_test - output_test.Y2).^2);
                % - Compute cost of all data:
                cost_LM(i+1,4) = 1/(2)*sum((Cm_all - output_all.Y2).^2);
                
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
        cost_fold(k,1) = mean(cost_init(:,1));
        cost_fold(k,2) = mean(cost_init(:,2));
        cost_fold(k,3) = mean(cost_init(:,3));
        cost_fold(k,4) = mean(cost_init(:,4));
    end
    
    cost_opt(n,1) = mean(cost_fold(:,1));
    cost_opt(n,2) = mean(cost_fold(:,2));
    cost_opt(n,3) = mean(cost_fold(:,3));
    cost_opt(n,4) = mean(cost_fold(:,4));
    
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
