function [net,E] = fixed_learning(func,net,input_train,input_test,Y_train,Y_test)

E = zeros(net.trainParam.epochs,2); % initialize the cost [cost_train cost_test] (this is per epoch)

lambda = 0.001;

% Initial value:
[net,E(1,:)] = func(net,input_train,input_test,Y_train,Y_test,lambda);
fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', 1, E(1,1), E(1,2), net.trainParam.mu);

for i = 2:net.trainParam.epochs  
    [net,E(i,:)] = func(net,input_train,input_test,Y_train,Y_test,lambda);  
    fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', i, E(i,1), E(i,2), net.trainParam.mu);
    %if E(i-1,:) < E(i,:)
    %     break;
    %end
end

end