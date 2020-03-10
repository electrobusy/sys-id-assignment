% Example with the cossine function in order to assess the previous
% networks

% Input data
x_train = 0:0.01:3*pi;
x_test = 0:0.009:3*pi;

% Output data - Expected
y_train = cos(x_train);
y_test = cos(x_test);

% figure(1);
% plot(x_train,y_train);
% xlabel('x');
% ylabel('y = cos(x)');
% legend('Expected');

% % DATA:
% Neural Net Structure parameters:
IN = 1;     % Fixed -> 1 inputs
HN = 40;    % Variable -> Neurons in the Hidden Layer
ON = 1;     % Fixed -> 1 output

mu = 1*10^-4; % Learning rate (initial one)
N_epochs = 200; % Number of Epochs

% Number of batches:
N_batches = 20; 


% Time to assess the network with the previously made functions:
% --> 1) Feedfoward:
% 1 - Create the net
net = createNet('feedforward',IN,HN,ON,mu,N_epochs);

% 2 - Get the inputs (for training and testing)
% -- training and testing data - decided in the beginning of the script

% 3 - Apply the learning algorithm
E = zeros(N_epochs,2);
alpha = 10;

% Compute the cost 1

for j = 1:N_epochs
    % Output:
    output = simNet(net,x_train,net.name);
    % Compute current cost:
    E(j,1) = 1/2*sum(y_train - output.Y2).^2;
    % Weight update:
    update = BackPropDerMat(net,output,x_train,y_train,0.4);
    % net_aux = weight_update(net_aux,update);
    % Compute later cost:
    % Compute training cost:
    output = simNet(net,x_train,net.name);
    E(j,1) = cost;
    % Compute testing cost:
    output = simNet(net,x_test,net.name);
    E(j,2) = 1/2*sum((output.Y2 - y_test).^2);
end

% 4 - Plots
% - Error:
figure();
plot(1:N_epochs,E(:,1),1:N_epochs,E(:,2));
xlabel('Epoch');
ylabel('Cost');
legend('train','test');
% - Function (Predicted + Real)
figure();
y_pred = simNet(net,x_test,net.name);
plot(x_train,y_train,'-',x_test,y_pred.Y2,'-.');


% -- Use adaptive learning algorithm:
% [net_LM,cost_LM] = adaptive_learning(@LM,net,input_train,input_test,Cm_train,Cm_test);
% 2 - Get the inputs
% -- all data
% input_ext = Z_k_rec_ext(:,[1 2])'; % [alpha beta] (filtered from KF) 
% input_ext_shuff = input_ext(:,index_shuff);
% -- training and testing data
% input_ext_train = input_ext(:,1:train_size); 
% input_ext_test = input_ext(:,train_size+1:end);
% 
% % -------------------- REDO!
% % 2.1 - Obtain output without any optimization
% % --> Output:
% % output_rbf_ext = simNet(net_ext,input_ext,'rbf'); % --> TEST
% % --> Plot:
% % plot_cm(output_rbf_ext.Y2,input_ext',num_fig);
% % 4 - Obtain the original function
% % --> Plot:
% % plot_cm(Cm,input_ext',num_fig+1);
% % --------------------
% 
% % 5 - Backpropagation:
% % N_batches = 1; % number of datapoints to be computed in order to update
% 
% % --> Obtain the the network with fixed-learning algorithm:
% % Acrescentar a parte de separar o data-set em treino e em teste!
% % [net_ext_back,cost_ext_back] = fixed_learning(@BackProp,net_ext,input_ext_train,input_ext_test,Cm_train,Cm_test);
% % figure()
% % plot(2:N_epochs,cost_ext_back(2:end,1),2:N_epochs,cost_ext_back(2:end,2));
% % legend('train','test');
% % output_rbf_ext_back = simNet(net_ext_back,input_ext,'rbf'); % --> TEST --> For all data points! 
% % plot_cm(output_rbf_ext_back.Y2',input_ext',num_fig+2);
% 
% % [net_ext_back_2,cost_ext_back_2] = adaptive_learning(@BackProp,net_ext,input_ext_train,input_ext_test,Cm_train,Cm_test,10);
% % figure();
% % plot(2:N_epochs,cost_ext_back_2(2:end,1),2:N_epochs,cost_ext_back_2(2:end,2));
% % legend('train','test');
% % output_rbf_ext_back_2 = simNet(net_ext_back_2,input_ext,'rbf'); % --> TEST --> For all data points! 
% % plot_cm(output_rbf_ext_back_2.Y2',input_ext',num_fig+3);
