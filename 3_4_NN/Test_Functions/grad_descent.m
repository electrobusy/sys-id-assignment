function net = grad_descent(net,update)

if strcmp(net.name,'feedforward')
    net.LW = net.LW - net.trainParam.mu*update.d_LW; 
    net.IW = net.IW - net.trainParam.mu*update.d_IW; 
    net.b{2} = net.b{2} - net.trainParam.mu*update.d_b_out; 
    net.b{1} = net.b{1} - net.trainParam.mu*update.d_b_in; 
elseif strcmp(net.name,'rbf') 
    net.LW = net.LW - net.trainParam.mu*update.d_LW;
    net.IW = net.IW - net.trainParam.mu*update.d_IW;
    net.centers = net.centers - net.trainParam.mu*update.d_centers;
end

end

% Stochastic:c
% Main loop for all training data points
    %for i = 1:length(Y_train)
        % - Get output
    %    output = simNet(net,input_train(:,i),net.name);
        % - Update network after N_batches:
        % if i == j
    %        net = LMDer(net,output,input_train(:,i),Y_train(i),lambda); % updates
        %    j = i + N_batches;
        %elseif i > floor(length(Y_train)/N_batches)*N_batches && i == length(Y_train) % last batch if not with N_batch elements 
        %    net = LMDer(net,output,input_train(:,i),Y_train(i),lambda); % update the last one
        %end
        % - Compute the training cost (for the learning algorithm)
    %    E_aux = 1/2*(Y_train(i) - output.Y2)^2; % desired - estimated
    %    E(1) = E(1) + E_aux;
    % end
    
        % Use entire data set: 
    % Main loop for all training data points
%    for i = 1:length(Y_train)
        % - Get output
%        output = simNet(net,input_train(:,i),net.name);
        % - For Ex. 4.2 - Update all the parameters - weights and bias
        % - Update network after N_batches:
        %if i == j
%            net = LMDer(net,output,input_train(:,i),Y_train(i),lambda); % updates
            %j = i + N_batches;
        %elseif i > floor(length(Y_train)/N_batches)*N_batches && i == length(Y_train) % last batch if not with N_batch elements 
        %    net = LMDer(net,output,input_train(:,i),Y_train(i),lambda); % update the last one
        %end
        % - Compute the training cost (for the learning algorithm)
%        E_aux = 1/2*(Y_train(i) - output.Y2)^2; % desired - estimated
%        E(1) = E(1) + E_aux;
%     end