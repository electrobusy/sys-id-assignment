% -- Includes extra code for batch learning (with mini-batches) and
% stochastic gradient descent. This script can be used to understand how
% these algorithms work.

% -- The code is messy, but can be adapted when compared to one of the
% algorithms presented in 3_2, 4_1 and 4_2. Due to time constraints, I did
% not do this, as this was also beyond of what was asked in the assignment.
% But the code presented below does work, only needs some adaptation!

% -- Take into account that these optimization methods take longer than the
% entire batch gradient descent that I made for the assignment. 

% Cheers,
% Rohan
% (C&S student)


%% Batch Learning part (updates with batches of data)
% -- Number elements per batch:
N_elem_per_batch = 100; % Batch learning -> Instead of learning with the 
                       % entire dataset, the training procedure can be 
                       % done with batches, i.e, update after x data
                       % points
                       
% % % ------------------------------------------------------------------------------------
% % % --> METHOD 2) ONLINE TRAINING ALGORITHM (TRAINING PER EXAMPLE) --> SIMILAR TO STOCHASTIC GRADIENT DESCENT
% % % ------------------------------------------------------------------------------------
% 
% % -- Recall the influence of the seed:
% rng(seed);
% 
% % -- Create the net
% net = createNet('feedforward',IN,HN,ON,mu,N_epochs);
% 
% % -- Learning rate increase or decrease: 
% alpha = 10;
% net.trainParam.mu_dec = 1/alpha;
% net.trainParam.mu_inc = alpha;
% 
% % -- Cost variable initialization: 
% cost_LM_2 = zeros(N_epochs+1,2); % [cost_train cost_test]
% 
% % -- Use adaptive learning learning algorithm:
% % - Forward prop:
% output_train = simNet(net,input_train,net.name);
% output_test = simNet(net,input_test,net.name);
% % - Compute current cost:
% cost_LM_2(1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
% cost_LM_2(1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
% % -- Auxiliary network:
% net_aux = net;
% 
% fprintf('METHOD 2\n');
% fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_2(1,1), cost_LM_2(1,2), net_aux.trainParam.mu);  
% 
% for i = 1:N_epochs 
%     % Shuffling the data at each epoch:
%     index_shuff = randperm(length(Cm)); % index for shuffling
%     Cm_shuff = Cm(index_shuff); % Cm is shuffled
%     Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled
%     
%     % Separate the training and testing data:
%     % -- Cm
%     Cm_train = Cm_shuff(1:train_size)'; % 80% training
%     Cm_test = Cm_shuff(train_size+1:end)'; % 20% testing
%     % -- input
%     input_train = Z_k_rec_shuff(1:train_size,[1 2])'; % 80% training
%     input_test = Z_k_rec_shuff(train_size+1:end,[1 2])'; % 20% testing
%     
%     % Training per data point
%     for j = 1:length(Cm_train)
%         % - Forward prop:
%         output_train = simNet(net_aux,input_train(:,j),net_aux.name);
%         % - Compute weight updates and update net:
%         [h_LM] = LMDer(net_aux,output_train,input_train(:,j),Cm_train(j),net_aux.trainParam.mu);
%         % - Update the network:
%         net_aux = LM_update(net_aux,h_LM);
%     end
%     
%     % -- Compute output after training the entire dataset:
%     % - Compute training output: 
%     output_train = simNet(net_aux,input_train,net_aux.name);
%     % - Compute testing output:
%     output_test = simNet(net_aux,input_test,net_aux.name);
%     
%     % - Compute current cost:
%     cost_LM_2(i+1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
%     cost_LM_2(i+1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
%     % % -----------------------------------------------------------------------------   
%     % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
%     if cost_LM_2(i+1,1) < cost_LM_2(i,1)
%         % Keep the changes: 
%         net = net_aux; 
%         % Increase learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
%     elseif cost_LM_2(i+1,1) >= cost_LM_2(i,1) 
%         % Save current mu to update:
%         mu_aux = net_aux.trainParam.mu;
%         % Reject the changes:
%         net_aux = net; % This net contains mu from the previous if
%         % Decrease learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_dec*mu_aux;
%         % Keep the previous cost function:
%         cost_LM_2(i+1,:) = cost_LM_2(i,:);
%     end
%     if sum(abs(h_LM)) < net_aux.trainParam.min_grad || (cost_LM_2(i+1,2) < net.trainParam.goal) || (net.trainParam.mu > 1e2)
%         break;
%     end
%     fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM_2(i+1,1), cost_LM_2(i+1,2), net_aux.trainParam.mu);
% end
% 
% % - Plots: 
% num_fig = num_fig + 1;
% figure(num_fig);
% semilogy(1:i,cost_LM_2(2:(i+1),1),1:i,cost_LM_2(2:(i+1),2));
% legend('Training error','Testing error');
% xlabel('Number of epochs [-]');
% ylabel('Cost [-]');
% title('Method 2 - Online training (batch of one sample)');
% 
% num_fig = num_fig + 1;
% net_output = simNet(net,net_input,net_aux.name);
% plot_cm(net_output.Y2,Z_k_rec(:,[1 2]),num_fig,'FF - Levenberg-Madquardt (Method 2)');
% 
% %% 
% % % -----------------------------------------------------------------------------
% % % --> METHOD 3) BATCH TRAINING ALGORITHM (BATCHES OF SAMPLES OF DATA) --> SIMILAR TO MINI-BATCH GRADIENT DESCENT
% % % -----------------------------------------------------------------------------
% 
% % -- Recall the influence of the seed: 
% rng(seed);
% 
% % -- Create the net
% net = createNet('feedforward',IN,HN,ON,mu,N_epochs);
% 
% % -- Learning rate increase or decrease: 
% alpha = 10;
% net.trainParam.mu_dec = 1/alpha;
% net.trainParam.mu_inc = alpha;
% 
% % -- Cost variable initialization: 
% cost_LM_3 = zeros(N_epochs+1,2); % [cost_train cost_test]
% 
% % -- Use adaptive learning learning algorithm:
% % - Forward prop:
% output_train = simNet(net,input_train,net.name);
% output_test = simNet(net,input_test,net.name);
% % - Compute current cost:
% cost_LM_3(1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
% cost_LM_3(1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
% % -- Auxiliary network:
% net_aux = net;
% 
% fprintf('METHOD 3\n');
% fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_3(1,1), cost_LM_3(1,2), net_aux.trainParam.mu);  
% 
% for i = 1:N_epochs 
%     % Shuffling the data:
%     index_shuff = randperm(length(Cm)); % indexes for shuffling
%     Cm_shuff = Cm(index_shuff); % Cm is shuffled
%     Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled
%     
%     % Separate the training and testing data:
%     % -- Cm
%     Cm_train = Cm_shuff(1:train_size)'; % 80% training
%     Cm_test = Cm_shuff(train_size+1:end)'; % 20% testing
%     % -- input
%     input_train = Z_k_rec_shuff(1:train_size,[1 2])'; % 80% training
%     input_test = Z_k_rec_shuff(train_size+1:end,[1 2])'; % 20% testing
%     
%     % Training per batch
%     for j = 1:floor(length(Cm_train)/N_elem_per_batch)
%         % - Input elements: 
%         k = ((j-1)*N_elem_per_batch + 1):j*N_elem_per_batch;
%         if j == floor(length(Cm_train)/N_elem_per_batch)
%             k = ((j-1)*N_elem_per_batch + 1):length(Cm_train);
%         end
%         % - Forward prop:
%         output_train = simNet(net_aux,input_train(:,k),net_aux.name);
%         % - Compute weight updates and update net:
%         [h_LM] = LMDerMat(net_aux,output_train,input_train(:,k),Cm_train(k),net_aux.trainParam.mu);
%         % - Update the network:
%         net_aux = LM_update(net_aux,h_LM);
%     end
%     
%     % -- Compute output after training the entire dataset:
%     % - Compute training output: 
%     output_train = simNet(net_aux,input_train,net_aux.name);
%     % - Compute testing output:
%     output_test = simNet(net_aux,input_test,net_aux.name);
%     
%     % -- Compute current cost:
%     cost_LM_3(i+1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
%     cost_LM_3(i+1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
%     
%     % % -----------------------------------------------------------------------------   
%     % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
%     if cost_LM_3(i+1,1) < cost_LM_3(i,1)
%         % Keep the changes: 
%         net = net_aux; 
%         % Increase learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
%     elseif cost_LM_3(i+1,1) >= cost_LM_3(i,1) 
%         % Save current mu to update:
%         mu_aux = net_aux.trainParam.mu;
%         % Reject the changes:
%         net_aux = net; % This net contains mu from the previous if
%         % Decrease learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_dec*mu_aux;
%         % Keep the previous cost function:
%         cost_LM_3(i+1,:) = cost_LM_3(i,:);
%     end
%     if (sum(abs(h_LM)) < net_aux.trainParam.min_grad) || (cost_LM_3(i+1,2) < net.trainParam.goal) || (net.trainParam.mu > 1e2)
%        break;
%     end
%     fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM_3(i+1,1), cost_LM_3(i+1,2), net_aux.trainParam.mu);
% end
% 
% % - Plots: 
% num_fig = num_fig + 1;
% figure(num_fig);
% semilogy(1:i,cost_LM_3(2:(i+1),1),1:i,cost_LM_3(2:(i+1),2));
% legend('Training error','Testing error');
% xlabel('Number of epochs [-]');
% ylabel('Cost [-]');
% title(['Method 3 - Batch training (batches of ' num2str(N_elem_per_batch) ' samples)']);
% 
% num_fig = num_fig + 1;
% net_output = simNet(net,net_input,net_aux.name);
% plot_cm(net_output.Y2,Z_k_rec(:,[1 2]),num_fig,'FF - Levenberg-Madquardt (Method 3) ');

%% 3.2 - LM
% % Batch Learning part (updates with batches of data)
% -- Number elements per batch:
N_elem_per_batch = 256;% Batch learning -> Instead of learning with the 
                       % entire dataset, the training procedure can be 
                       % done with batches, i.e, update after x data
                       % points
                       
% %%
% % % ------------------------------------------------------------------------------------
% % % --> METHOD 2) ONLINE TRAINING ALGORITHM (TRAINING PER EXAMPLE) --> SIMILAR TO STOCHASTIC GRADIENT DESCENT
% % % ------------------------------------------------------------------------------------
% 
% % -- Recall the influence of the seed:
% rng(seed);
% 
% % -- Create the net
% net = createNet('rbf',IN,HN,ON,mu,N_epochs);
% 
% % -- Learning rate increase or decrease: 
% alpha = 10;
% net.trainParam.mu_dec = 1/alpha;
% net.trainParam.mu_inc = alpha;
% 
% % -- Cost variable initialization: 
% cost_LM_2 = zeros(N_epochs+1,2); % [cost_train cost_test]
% 
% % -- Use adaptive learning learning algorithm:
% % - Forward prop:
% output_train = simNet(net,input_train,net.name);
% output_test = simNet(net,input_test,net.name);
% % - Compute current cost:
% cost_LM_2(1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
% cost_LM_2(1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
% % -- Auxiliary network:
% net_aux = net;
% 
% fprintf('METHOD 2\n');
% fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_2(1,1), cost_LM_2(1,2), net_aux.trainParam.mu);  
% 
% for i = 1:N_epochs 
%     % Shuffling the data at each epoch:
%     index_shuff = randperm(length(Cm)); % index for shuffling
%     Cm_shuff = Cm(index_shuff); % Cm is shuffled
%     Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled
%     
%     % Separate the training and testing data:
%     % -- Cm
%     Cm_train = Cm_shuff(1:train_size)'; % 80% training
%     Cm_test = Cm_shuff(train_size+1:end)'; % 20% testing
%     % -- input
%     input_train = Z_k_rec_shuff(1:train_size,[1 2])'; % 80% training
%     input_test = Z_k_rec_shuff(train_size+1:end,[1 2])'; % 20% testing
%     
%     % Training per data point
%     for j = 1:length(Cm_train)
%         % - Forward prop:
%         output_train = simNet(net_aux,input_train(:,j),net_aux.name);
%         % - Compute weight updates and update net:
%         [h_LM] = LMDer(net_aux,output_train,input_train(:,j),Cm_train(j),net_aux.trainParam.mu);
%         % - Update the network:
%         net_aux = LM_update(net_aux,h_LM);
%     end
%     
%     % -- Compute output after training the entire dataset:
%     % - Compute training output: 
%     output_train = simNet(net_aux,input_train,net_aux.name);
%     % - Compute testing output:
%     output_test = simNet(net_aux,input_test,net_aux.name);
%     
%     % - Compute current cost:
%     cost_LM_2(i+1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
%     cost_LM_2(i+1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
%     % % -----------------------------------------------------------------------------   
%     % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
%     if cost_LM_2(i+1,1) < cost_LM_2(i,1)
%         % Keep the changes: 
%         net = net_aux; 
%         % Increase learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
%     elseif cost_LM_2(i+1,1) >= cost_LM_2(i,1) 
%         % Save current mu to update:
%         mu_aux = net_aux.trainParam.mu;
%         % Reject the changes:
%         net_aux = net; % This net contains mu from the previous if
%         % Decrease learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_dec*mu_aux;
%         % Keep the previous cost function:
%         cost_LM_2(i+1,:) = cost_LM_2(i,:);
%     end
%     if sum(abs(h_LM)) < net_aux.trainParam.min_grad || (cost_LM_2(i+1,2) < net.trainParam.goal) || (net.trainParam.mu > 1e2)
%         break;
%     end
%     fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM_2(i+1,1), cost_LM_2(i+1,2), net_aux.trainParam.mu);
% end
% 
% % - Plots: 
% num_fig = num_fig + 1;
% figure(num_fig);
% semilogy(1:i,cost_LM_2(2:(i+1),1),1:i,cost_LM_2(2:(i+1),2));
% legend('Training error','Testing error');
% xlabel('Number of epochs [-]');
% ylabel('Cost [-]');
% title('Method 2 - Online training (batch of one sample)');
% 
% num_fig = num_fig + 1;
% net_output = simNet(net,net_input,net_aux.name);
% plot_cm(net_output.Y2,Z_k_rec(:,[1 2]),num_fig,'RBF - Levenberg-Madquardt (Method 2)');

% %% 
% % % -----------------------------------------------------------------------------
% % % --> METHOD 3) BATCH TRAINING ALGORITHM (BATCHES OF SAMPLES OF DATA) --> SIMILAR TO MINI-BATCH GRADIENT DESCENT
% % % -----------------------------------------------------------------------------
% 
% % -- Recall the influence of the seed: 
% rng(seed);
% 
% % -- Create the net
% net = createNet('rbf',IN,HN,ON,mu,N_epochs);
% 
% % -- Learning rate increase or decrease: 
% alpha = 10;
% net.trainParam.mu_dec = 1/alpha;
% net.trainParam.mu_inc = alpha;
% 
% % -- Cost variable initialization: 
% cost_LM_3 = zeros(N_epochs+1,2); % [cost_train cost_test]
% 
% % -- Use adaptive learning learning algorithm:
% % - Forward prop:
% output_train = simNet(net,input_train,net.name);
% output_test = simNet(net,input_test,net.name);
% % - Compute current cost:
% cost_LM_3(1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
% cost_LM_3(1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
% 
% % -- Auxiliary network:
% net_aux = net;
% 
% fprintf('METHOD 3\n');
% fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', 0, cost_LM_3(1,1), cost_LM_3(1,2), net_aux.trainParam.mu);  
% 
% for i = 1:N_epochs 
%     % Shuffling the data:
%     index_shuff = randperm(length(Cm)); % indexes for shuffling
%     Cm_shuff = Cm(index_shuff); % Cm is shuffled
%     Z_k_rec_shuff = Z_k_rec(index_shuff,:); % alpha and beta should also be shuffled
%     
%     % Separate the training and testing data:
%     % -- Cm
%     Cm_train = Cm_shuff(1:train_size)'; % 80% training
%     Cm_test = Cm_shuff(train_size+1:end)'; % 20% testing
%     % -- input
%     input_train = Z_k_rec_shuff(1:train_size,[1 2])'; % 80% training
%     input_test = Z_k_rec_shuff(train_size+1:end,[1 2])'; % 20% testing
%     
%     % Training per batch
%     for j = 1:floor(length(Cm_train)/N_elem_per_batch)
%         % - Input elements: 
%         k = ((j-1)*N_elem_per_batch + 1):j*N_elem_per_batch;
%         if j == floor(length(Cm_train)/N_elem_per_batch)
%             k = ((j-1)*N_elem_per_batch + 1):length(Cm_train);
%         end
%         % - Forward prop:
%         output_train = simNet(net_aux,input_train(:,k),net_aux.name);
%         % - Compute weight updates and update net:
%         [h_LM] = LMDerMat(net_aux,output_train,input_train(:,k),Cm_train(k),net_aux.trainParam.mu);
%         % - Update the network:
%         net_aux = LM_update(net_aux,h_LM);
%     end
%     
%     % -- Compute output after training the entire dataset:
%     % - Compute training output: 
%     output_train = simNet(net_aux,input_train,net_aux.name);
%     % - Compute testing output:
%     output_test = simNet(net_aux,input_test,net_aux.name);
%     
%     % - Compute current cost:
%     cost_LM_3(i+1,1) = (1/(2*length(Cm_train)))*sum((Cm_train - output_train.Y2).^2); % training cost
%     cost_LM_3(i+1,2) = (1/(2*length(Cm_test)))*sum((Cm_test - output_test.Y2).^2); % testing cost
%     
%     % % -----------------------------------------------------------------------------   
%     % % -- EVALUATION OF THE COST FUNCTION AFTER TRAINING ENTIRE BATCH  
%     if cost_LM_3(i+1,1) < cost_LM_3(i,1)
%         % Keep the changes: 
%         net = net_aux; 
%         % Increase learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_inc*net_aux.trainParam.mu;
%     elseif cost_LM_3(i+1,1) >= cost_LM_3(i,1) 
%         % Save current mu to update:
%         mu_aux = net_aux.trainParam.mu;
%         % Reject the changes:
%         net_aux = net; % This net contains mu from the previous if
%         % Decrease learning rate:
%         net_aux.trainParam.mu = net_aux.trainParam.mu_dec*mu_aux;
%         % Keep the previous cost function:
%         cost_LM_3(i+1,:) = cost_LM_3(i,:);
%     end
%     if (sum(abs(h_LM)) < net_aux.trainParam.min_grad) || (cost_LM_3(i+1,2) < net.trainParam.goal) || (net.trainParam.mu > 1e2)
%        break;
%     end
%     fprintf('Epoch %d: Training error = %f / Testing Error = %f / Learning rate = %f \n', i, cost_LM_3(i+1,1), cost_LM_3(i+1,2), net_aux.trainParam.mu);
% end
% 
% % - Plots: 
% num_fig = num_fig + 1;
% figure(num_fig);
% semilogy(1:i,cost_LM_3(2:(i+1),1),1:i,cost_LM_3(2:(i+1),2));
% legend('Training error','Testing error');
% xlabel('Number of epochs [-]');
% ylabel('Cost [-]');
% title(['Method 3 - Batch training (batches of ' num2str(N_elem_per_batch) ' samples)']);
% 
% num_fig = num_fig + 1;
% net_output = simNet(net,net_input,net_aux.name);
% plot_cm(net_output.Y2,Z_k_rec(:,[1 2]),num_fig,'RBF - Levenberg-Madquardt (Method 3) ');