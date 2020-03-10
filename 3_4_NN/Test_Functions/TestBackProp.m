% Test BackProp

clear 
close all

load('reconstructed_data_KF.mat');

% Figure number:
num_fig = 1;

% Neural Net Structure parameters:
IN = 2;     % Fixed -> 2 inputs
HN = 50;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^-3; % Learning rate (initial one)
N_epochs = 200; % Number of Epochs

% Let us just worry about the Z_k_rec_ext = [alpha beta V] (FOR NOW)

alpha = Z_k_rec_ext(:,1);
beta = Z_k_rec_ext(:,2);

data_pts = [alpha beta Cm];

% Define training and test set
% --> First shuffle the elements:
index_shuf = randperm(size(data_pts,1));
data_pts_shuf = data_pts(index_shuf,:);

% --> Create training and validation set:
training_percentage = 0.8;
training_length = floor(training_percentage*size(data_points,1));
d_pts_train = data_pts_shuf(1:training_length,:);
d_pts_test = data_pts_shuf(training_length+1:end,:);

% 1 - Create the net
net_ext = createNet('rbf',IN,HN,ON,mu,N_epochs);

% 2 - Get the inputs
input_ext_train = d_pts_train([1 2],:)'; % [alpha beta] (filtered from KF)
input_ext_test = d_pts_test([1 2],:)';

% 3 - Fixed Learning algorithms

E = zeros(net_ext.trainParam.epochs,1); % initialize the cost

% Training part: 
% 1 - compute cost function for the current cost function E_t
% 1b - compute weight update for the current set of weights W_t 
% and
% 2a - Perform update of the weights
[net_aux,E(1)] = BackProp(net_ext,N_batches,input,Y,0.4); 

for i = 2:net.trainParam.epochs  
    % 2b - Compute new cost function value Et+1
    [net_aux,E(i)] = func(net_aux,N_batches,input,Y,0.4); 
    % 3a - If E_(t+1) < E_(t) then accept changes 
    if E(i) < E(i-1)
        net_upd = net_aux;
    % 3b - else stop loop
    else
        break;
    end      
end


