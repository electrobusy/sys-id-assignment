function [net_upd,E] = adaptive_learning(func,net,input_train,input_test,Y_train,Y_test,alpha)

E = zeros(net.trainParam.epochs,2); % initialize the cost [cost_train cost_test] (this is per epoch)

% Dependence of the learning rate with alpha
net.trainParam.mu_dec = 1/alpha;
net.trainParam.mu_inc = alpha;

mu = zeros(1,net.trainParam.epochs);

% Training part: 
% 1 - compute cost function for the current cost function E_t
% 1b - compute weight update for the current set of weights W_t 
% and
% 2a - Perform update of the weights
[net_aux,E(1,:)] = func(net,input_train,input_test,Y_train,Y_test,0.4); 
mu(1) = net_aux.trainParam.mu;

for i = 2:net.trainParam.epochs   
    % 2b - Compute new cost function value Et+1
    [net_aux,E(i,:)] = func(net_aux,input_train,input_test,Y_train,Y_test,0.4);
    % 3a - If Et+1 < Et then accept changes and increase learning rate
    if E(i,1) < E(i-1,1)
        net_upd = net_aux;
        net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
    % 3b - else decrease learning rate
    else
        net_aux.trainParam.mu = net_aux.trainParam.mu_dec*net_aux.trainParam.mu;
    end  
    mu(i) = net_aux.trainParam.mu;
    % 3c - partial derivatives are (nearly) zero
    % if abs(update(i)) < net.min_grad % [CHANGE THIS PART] - ver
    %    break;
    % end
end

figure(4);
plot(1:net.trainParam.epochs,mu);

end