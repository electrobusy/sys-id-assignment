% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: NN_LM_Adaptive_OPT.m

clear all 

%% DATA: -- Simulated data

dt = 0.05;
x = (0:dt:2*pi)';
func = x.^2; % -- choose the function you would like to approximate
y = func + 0.1*rands(length(x),1);

%% Define the seed (using random functions)
% seed = 123456;
% rng(seed);

%% Paramaters of the NN:

% % Neural Net Structure parameters:
IN = 1;     % Fixed -> one inputs
HN = 15;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> one output

mu = 1*10^3; % Learning rate (initial one)
N_epochs = 4000; % Number of Epochs

alpha = 10; % Incremental/decremental factor for the learning rate

% Weight initializations
num_init = 10;

% % Max number of neurons to test - Sensibility and optimization part
max_HN = 20;

%% Data management (later for the LM algorithm):

% % - Split dataset in Training and Testing data
% -- METHOD: Use the dividerand function
trainRatio = 0.3;
valRatio = 0;
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

%% OPTIMIZATION WITH THE BACKPROP ALGORITHM:

% -- Optimal cost variable initialization: 
cost_opt = zeros(max_HN,4); % [cost_train cost_val cost_test cost_all]

% -- Cost after training for each initialization:
cost_init = zeros(num_init,4); % [cost_train cost_val cost_test cost_all]

for n = 1:max_HN

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
        output_all = simNet(net,x',net.name);

        % - Compute current training/validation/testing cost:
        cost_LM(1,1) = 1/(2)*sum((y_train - output_train.Y2).^2); % training cost
        cost_LM(1,2) = 1/(2)*sum((y_val - output_val.Y2).^2); % testing cost
        cost_LM(1,3) = 1/(2)*sum((y_test - output_test.Y2).^2); % testing cost
        cost_LM(1,4) = 1/(2)*sum((y' - output_all.Y2).^2); % all data cost

        % -- Auxiliary network:
        net_aux = net;
        
        % fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f / Learning rate = %f \n', 0, cost_LM(1,1), cost_LM(1,2), cost_LM(1,3), cost_LM(1,4), net_aux.trainParam.mu);
        
        for i = 1:N_epochs
            % - Forward prop:
            output_train = simNet(net_aux,input_train,net.name);
            
            % - Backpropagation and network update:
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
            % - Computer all data output:
            output_all = simNet(net_aux,x',net.name);
            
            % - Compute training cost:
            cost_LM(i+1,1) = 1/(2)*sum((y_train - output_train.Y2).^2);
            % - Compute validation cost:
            cost_LM(i+1,2) = 1/(2)*sum((y_val - output_val.Y2).^2);
            % - Compute testing cost:
            cost_LM(i+1,3) = 1/(2)*sum((y_test - output_test.Y2).^2);
            % - Compute cost of all data:
            cost_LM(i+1,4) = 1/(2)*sum((y' - output_all.Y2).^2); 
            if isnan(cost_LM(i+1,1)) == 1
                disp('yo');
            end
            % % -----------------------------------------------------------------------------
            % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH
            if cost_LM(i+1,1) <= cost_LM(i,1)
                % Keep the changes:
                net = net_aux;
                % Increase learning rate:
                net_aux.trainParam.mu = net_aux.trainParam.mu_dec*net_aux.trainParam.mu;
            elseif cost_LM(i+1,1) > cost_LM(i,1)
                % Save current mu to update:
                mu_aux = net_aux.trainParam.mu;
                % Reject the changes:
                net_aux = net; % This net contains mu from the previous if
                % Decrease learning rate:
                net_aux.trainParam.mu = net_aux.trainParam.mu_inc*mu_aux;
                % Keep the previous cost function:
                cost_LM(i+1,:) = cost_LM(i,:);
            end
            if cost_LM(i+1,1) < net_aux.trainParam.goal || abs(min(grad)) < net_aux.trainParam.min_grad  || net_aux.trainParam.mu > net.trainParam.mu_max
                % cost_LM(i+1,:) = cost_LM(i,:);
                break;
            end
            % fprintf('Epoch %d: Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f / Learning rate = %f \n', i, cost_LM(i+1,1), cost_LM(i+1,2), cost_LM(i+1,3), cost_LM(i+1,4), net_aux.trainParam.mu);
        end
        cost_init(j,1) = cost_LM(i+1,1);
        cost_init(j,2) = cost_LM(i+1,2);
        cost_init(j,3) = cost_LM(i+1,3);
        cost_init(j,4) = cost_LM(i+1,4);
    end
    cost_opt(n,1) = mean(cost_init(:,1));
    cost_opt(n,2) = mean(cost_init(:,2));
    cost_opt(n,3) = mean(cost_init(:,3));
    cost_opt(n,4) = mean(cost_init(:,4));
    fprintf('# of Neurons = %d : Training error = %f / Validation Error = %f / Testing Error = %f / All data Error: %f \n', n, cost_opt(n,1), cost_opt(n,2), cost_opt(n,3), cost_opt(n,4));
end

% % Plots:
num_fig = 1;
figure(num_fig);
set(gca, 'YScale', 'log')
semilogy(1:max_HN,cost_opt(:,1),1:max_HN,cost_opt(:,2),1:max_HN,cost_opt(:,3),1:max_HN,cost_opt(:,4));
legend('Training','Validation','Testing','All data');
xlabel('Number of neurons [-]');
ylabel('Cost after training [-]');
title('Optimization of number of neurons');
