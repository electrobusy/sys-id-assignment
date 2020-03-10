% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: RBF_ex_3_3_OLS_cross_val.m

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

%% Load functions from parameter estimator (Use the OLS algorithm):

addpath(genpath('../2_Par_Est'));

%% Define the seed (using random functions)
seed = 123456;
rng(seed);

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 2;     % Fixed -> 2 inputs
HN = 10;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^3; % Learning rate (initial one)
N_epochs = 250; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

% Number of weight initializations
num_init = 5;

% % Max number of neurons to test - Sensibility and optimization part
max_HN = 35;

%% Data management (later for the algorithm):

% % Shuffling the data at each epoch:
index_shuff = randperm(length(Cm)); % index for shuffling
Cm_shuff = Cm(index_shuff); % Cm is shuffled
Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled

% % - Training/Valitation/Testing data:
train_pct = 0.8; % percentage of training data to be used
val_pct = 0.1; % percentage of validation data to be used

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
            
            % % 1 - Create the net
            net = createNet('rbf',IN,n,ON,mu,N_epochs);
            
            % % 2 - [OPTIONAL, BUT RECOMENDED FROM LITERATURE]
            % RBFs locations - using k_means clustering:
            [~,loc_cluster] = kmeans(input_train',n);
            
            % Since the neurons are defined as a 3-D Normal Distribution, c_ij of the
            % neural net are changed and they are given the values of loc_cluster:
            net.centers = loc_cluster;
            
            % % 3 - Use OLS algorithm to optimize the outer-most weights of the network
            % using the given data:
            % -- Initialize the regression matrix:
            output_train = simNet(net,input_train,'rbf');
            A_train = output_train.Y1';
            % -- Use the OLS algorithm:
            [ls_par_rbf_OLS,Y_rbf_OLS_train] = ord_least_squares(A_train,Cm_train');

            % % -- Determine the regression matrix for validation/testing
            % and entire dataset:
            % - Forward prop:
            output_val = simNet(net,input_val,net.name);
            output_test = simNet(net,input_test,net.name);
            output_all = simNet(net,input_all,net.name);
            % - Regression matrix:
            A_val = output_val.Y1';
            A_test = output_test.Y1';
            A_all = output_all.Y1';
            
            % % - Determine the output for the validation/testing and
            % entire data set:
            Y_rbf_OLS_val = A_val*ls_par_rbf_OLS;
            Y_rbf_OLS_test = A_test*ls_par_rbf_OLS;
            Y_rbf_OLS_all = A_all*ls_par_rbf_OLS;
            
            % - Compute current training/validation/testing cost:
            cost_init(j,1) = 1/(2)*sum((Cm_train - Y_rbf_OLS_train').^2); % training cost
            cost_init(j,2) = 1/(2)*sum((Cm_val - Y_rbf_OLS_val').^2); % testing cost
            cost_init(j,3) = 1/(2)*sum((Cm_test - Y_rbf_OLS_test').^2); % testing cost
            cost_init(j,4) = 1/(2)*sum((Cm_all - Y_rbf_OLS_all').^2); % all data cost
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
yL = get(gca,'YLim'); % Optimal number of neurons -- RBF and FF
p2= line([idx_HN idx_HN],yL,'Linestyle','-.','Color','b',...
    'Linewidth',1);
xL = get(gca,'XLim'); % Optimal cost -- RBF and FF
p1 = line(xL,[cost_opt(idx_HN,4) cost_opt(idx_HN,4)],'Linestyle','-.','Color','r',...
    'Linewidth',1);
legend('Training','Validation','Testing','All data',['Optimal HN = ' num2str(idx_HN)],['Optimal cost = ' num2str(cost_opt(idx_HN,4))]);
xlabel('Number of neurons [-]');
ylabel('Cost after training [-]');
title('Optimization of number of neurons');