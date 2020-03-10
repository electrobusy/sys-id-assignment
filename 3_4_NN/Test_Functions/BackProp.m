function [net_upd,E] = BackProp(net,input_train,input_test,Y_train,Y_test,lambda) 
% NOTE: Lambda is not used -> it will be used in the LM algorithm, but the
% function should have this input to work later.

E = [0 0]; % [train_cost test_cost]

% - Variable needed in the implementation of batches
% j is a variable needed to update the Network
% j = 2 + N_batches - 1;

% --> FeedForward Network -> Ex. 4
if strcmp(net.name,'feedforward')
    % % TRAINING PHASE
    % - Get the first output:
    output = simNet(net,input_train(:,1),net.name);
    [update] = BackPropDer(net,output,input_train(:,1),Y_train(1),lambda); % updates
    E_train(1) = 1/2*sum((Y_train - output.Y2).^2);
    % Main loop for all training data points
    for i = 2:length(Y_train)
        
        net.LW = net.LW - net.trainParam.mu*update.d_LW;
        net.IW = net.IW - net.trainParam.mu*update.d_IW;
        net.b{2,1} = net.b{2,1} - net.trainParam.mu*update.d_b_out;
        net.b{1,1} = net.b{1,1} - net.trainParam.mu*update.d_b_in;
        
        % - Get output
        output = simNet(net,input_train(:,i),net.name);
        E_train(i) = 1/2*sum((Y_train(i) - output.Y2).^2);
        % - Update network after N_batches:
        %if i == j
            [update] = BackPropDer(net,output,input_train(:,i),Y_train(i),lambda); % updates
            %j = i + N_batches;
        %elseif i > floor(length(Y_train)/N_batches)*N_batches && i == length(Y_train) % last batch if not with N_batch elements 
        %    update = BackPropDer(net,output,input_train(:,i),Y_train(i),lambda); % update the last one
        %end
        % - For Ex. 4.1 - Update all the parameters - weights and bias
        % net.LW = net.LW - net.trainParam.mu*update.d_LW;
        % net.IW = net.IW - net.trainParam.mu*update.d_IW;
        % net.b{2,1} = net.b{2,1} - net.trainParam.mu*update.d_b_out;
        %   net.b{1,1} = net.b{1,1} - net.trainParam.mu*update.d_b_in;
        
        % 
        % output_hat = simNet(net,input_train(:,i),net.name);
        % [e_hat,~] = BackPropDer(net,output_hat,input_train(:,i),Y_train(i),lambda); % updates
        
        if abs(E_train(i)) < abs(E_train(i-1))
            net_upd = net;
            net.trainParam.mu = net.trainParam.mu*10;
        else
            net.trainParam.mu = net.trainParam.mu*10^-1;
        end
            
    end
% --> RBF Network -> Ex. 3
elseif strcmp(net.name,'rbf')    
    % % TRAINING PHASE
    % Main loop for all training data points
    for i = 1:length(Y_train)
        % - Get output
        output = simNet(net,input_train(:,i),net.name);
        % - Update network after N_batches:
        % if i == j
            [e,update] = BackPropDer(net,output,input_train(:,i),Y_train(i),lambda); % updates
        %    j = i + N_batches;
        %elseif i > floor(length(Y_train)/N_batches)*N_batches && i == length(Y_train) % last batch if not with N_batch elements 
        %    update = BackPropDer(net,output,input_train(:,i),Y_train(i),lambda); % update the last one
        %end
        % - For Ex. 3.1 - Only optimize/update w_j = a
        net.LW = net.LW - net.trainParam.mu*update.d_LW; % [ASKED IN 3.1]
        % - For Ex. 3.2 - Only optimize/update w_ij and centers of RBF
        % net.IW = net.IW - net.trainParam.mu*update.d_IW; [NOT ASKED IN BACKPROP]
        % net.centers = net.centers - net.trainParam.mu*update.d_centers; [NOT ASKED IN BACKPROP]
    end
% If the network is not found 
else
    fprintf('<BackProp.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
    return;
end

% Update Network:
% net_upd = net_aux;

% - Compute the training cost (for the learning algorithm) ->  for the entire training set
output = simNet(net_upd,input_train,net_upd.name);
E(1) = 1/2*sum((Y_train - output.Y2).^2);
    
% % TESTING PHASE
% - Compute the testing cost (for the learning algorithm)
output = simNet(net_upd,input_test,net_upd.name);
E(2) = 1/2*sum((Y_test - output.Y2).^2);

end