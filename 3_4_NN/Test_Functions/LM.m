function [net,E] = LM(net,input_train,input_test,Y_train,Y_test,lambda)

E = [0 0]; % [train_cost test_cost]

% --> FeedForward Network -> Ex. 4
if strcmp(net.name,'feedforward')
    % % TRAINING PHASE
    % -- Gradient Descent: Update entire data-set at once - one weight-update per epoch 
    % update = LMDerMat(net,output,input,Y,lambda);
    % -- Stochastic Gradient Descent: Update per training example - many weight-updates per epoch
    
    % -- Mini-Batch Gradient Descent: Update per batch - many weight-updates per epoch
    
% --> RBF Network -> Ex. 3
else if strcmp(net.name,'rbf')    
    % % TRAINING PHASE
    % -- Gradient Descent: Update entire data-set at once - one weight-update per epoch 
    output = simNet(net,input_train,net.name); % get output of entire dataset
    net = LMDer(net,output,input_train,Y_train,lambda);
    % -- Stochastic Gradient Descent: Update per training example - many weight-updates per epoch
    
    % -- Mini-Batch Gradient Descent: Update per batch - many weight-updates per epoch
    
    % % Compute the cost:  
    E(1) = 1/2*sum((Y_train - output.Y2).^2);
    
% If the network is not found 
else
    fprintf('<LM.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
    return;
    end
    
% % TESTING PHASE
% - Compute the testing cost (for the learning algorithm)
output = simNet(net,input_test,net.name);
E(2) = (1/2)*sum((Y_test - output.Y2).^2);

end