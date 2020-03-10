% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: RBF_ex_3_1.m

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
HN = 24;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^-3; % Learning rate (initial one)
N_epochs = 10; % Number of Epochs

% -- Maximum number of neurons to test --> Sensitivity Analysis
max_HN = 100;

%% Ex_3_1 - Linear Regression approach to NN:

% % 1 - Create the net
net = createNet('rbf',IN,HN,ON,mu,N_epochs);

% % 2 - [OPTIONAL, BUT RECOMENDED FROM LITERATURE] 
% RBFs locations - using k_means clustering:
[~,loc_cluster] = kmeans(Z_k_rec(:,[1 2]),HN); 

% Since the neurons are defined as a 3-D Normal Distribution, c_ij of the
% neural net are changed and they are given the values of loc_cluster:
net.centers = loc_cluster;

% % 3 - Use OLS algorithm to optimize the outer-most weights of the network
% using the given data:
% Transpose the data input and give it to another variable (more readable):
net_input = Z_k_rec(:,[1 2])';
% -- Initialize the regression matrix:
net_output = simNet(net,net_input,'rbf');
A = net_output.Y1';
% -- Use the OLS algorithm:
[ls_par_rbf_OLS,Y_rbf_OLS] = ord_least_squares(A,Cm);

% % 4 - Plot:
% Figure number:
num_fig = 1;
plot_Cm(Cm,Y_rbf_OLS,Z_k_rec(:,[1 2]),num_fig,['RBF-NN + OLS - ' data_set ' -> Reconstructed C_m for ' num2str(HN) ' neurons']);
num_fig = num_fig + 1;
plot_RBF_inputs(Cm,num_fig,['RBF-NN + OLS - ' data_set ' -> For ' num2str(HN) ' neurons'],Z_k_rec(:,[1 2]),net.centers);

cost_init = 1/(2)*sum((Cm - Y_rbf_OLS).^2); 

fprintf('# of neurons = %d / Cost = %f\n', HN, cost_init);

% % 5 - [IMPORTANT] Sensitivity Analysis - analyse design choices:
% -- Uniform distribution from [-x,x] for inner weights:
x = 30;

% -- Cost variable initialization:
cost = zeros(max_HN,x);
cost_min = zeros(max_HN,1); % minimum value of cost function 
cost_idx = zeros(max_HN,1); % idx represents the x in for the uniform interval

% -- Optimization loop:
for j = 1:max_HN
    % -- Keep the seed constant in order to initialize the weights
    % consistently:
    rng(seed);
    
    % -- Initialize the net:
    net_aux = createNet('rbf',IN,j,ON,mu,N_epochs);
    
    % -- Use k-means to organize data since it takes into account the number 
    % of neurons:
    [~,loc_cluster] = kmeans(Z_k_rec(:,[1 2]),j);
    
    % -- Change location of RBF neurons using what was obtained from
    % k-means:
    net_aux.centers = loc_cluster;
    
    % -- Neuron's width analysis:
    for i = 1:x
        % -- Change the neurons width - inner weights:
        net_aux.IW = -i + (i-(-i)).*rand(j,IN);
        % -- Use the OLS algorithm:
        net_output = simNet(net_aux,net_input,'rbf');
        A_aux = net_output.Y1';
        [ls_par_rbf_OLS,Y_rbf_OLS] = ord_least_squares(A_aux,Cm);
        % -- Assess the network with the total cost function:
        cost(j,i) = 1/(2)*sum((Cm - Y_rbf_OLS).^2);  
    end
    
    % -- Determine the minimum cost function and its index:
    [cost_min(j),cost_idx(j)] = min(cost(j,:));  
end

num_fig = num_fig + 1;
figure(num_fig);

suptitle(['RBF-NN + OLS - ' data_set ' - Sensitivity Analysis']);
subplot(2,1,1);
plot(1:max_HN,cost_min);
xlabel('Number of Neurons');
ylabel('Minimum Cost')

subplot(2,1,2);
plot(1:max_HN,cost_idx);
xlabel('Number of Neurons');
ylabel('x - Uniform Dist. Interval');

% Observing the trend, the best values appear to be when there are more
% neurons and their width is wider. Considering the last values obtained
% from the optimization algorithm: 
num_fig = num_fig + 1;
plot_Cm(Cm,Y_rbf_OLS,Z_k_rec(:,[1 2]),num_fig,['RBF-NN + OLS - ' data_set ' -> Reconstructed C_m for ' num2str(j) ' neurons']);
num_fig = num_fig + 1;
plot_RBF_inputs(Cm,num_fig,['RBF-NN + OLS - ' data_set ' -> For ' num2str(j) ' neurons'],Z_k_rec(:,[1 2]),net_aux.centers);

