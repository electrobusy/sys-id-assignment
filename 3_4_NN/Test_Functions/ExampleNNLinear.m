% Example with simple linear function in ordere to assess the FF NN

% Input data:
% - train
x_train = 0:0.001:1;
y_train = x_train;
% - test
x_test = 0:0.006:1;
y_test = x_test;

% figure(1);
% plot(x_train,y_train);
% xlabel('x');
% ylabel('y = x');
% legend('Expected');

% % Neural Net:
% Neural Net Structure parameters:
IN = 1;     % Fixed -> 1 inputs
HN = 20;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> 1 output

mu = 1*10^-4; % Learning rate (initial one)
N_epochs = 100000; % Number of Epochs

% Number of batches:
N_batches = length(x_train); 

% Time to assess the network with the previously made functions:
% --> 1) Feedfoward:
% 1 - Create the net
net = createNet('feedforward',IN,HN,ON,mu,N_epochs);

% 2 - Get the inputs (for training and testing)
% -- training and testing data - decided in the beginning of the script

% 3 - Apply the learning algorithm
E = zeros(N_epochs,2);
alpha = 10;

% Output:
output_train = simNet(net,x_train,net.name);

% Compute current cost (training cost)
E(1,1) = 1/2*sum(y_train - output_train.Y2).^2;

% Compute testing cost:
output_test = simNet(net,x_test,net.name);
E(1,2) = 1/2*sum((output_test.Y2 - y_test).^2);

for j = 2:N_epochs
    % Weight update:
    update = BackPropDerMat(net,output_train,x_train,y_train,0.4);
    net = grad_descent(net,update);
    % Compute new output and cost:
    output_train = simNet(net,x_train,net.name);
    E(j,1) = 1/2*sum(y_train - output_train.Y2).^2;
    if E(j,1) < E(j-1,1)
        % net = net_aux;
        % Compute testing cost:
        output_test = simNet(net,x_test,net.name);
        E(j,2) = 1/2*sum((output_test.Y2 - y_test).^2);
        fprintf('Esta merda funciona?!\n');
    else E(j,1) > E(j-1,1)
        
        break;
    end
end

% 4 - Plots
% - Error:
figure(2);
plot(1:N_epochs,E(:,1),1:N_epochs,E(:,2));
xlabel('Epoch');
ylabel('Cost');
legend('train','test');
% - Function (Predicted + Real)
figure(3);
y_pred = simNet(net,x_test,net.name);
plot(x_train,y_train,'-',x_test,y_pred.Y2,'-.');