function [h_LM,grad] = LMDerMat(net,output,input,Y,lambda)

% --> FeedForward Network -> Ex. 4
if strcmp(net.name,'feedforward')    
    % output bias update - b_out
    d_b_out = (-1); 
    
    % output layer weight update - LW
    d_LW = (d_b_out.*output.Y1)';
    
    % input bias update - b_int
    aux = exp(-2.*output.V1);
    d_aux = d_b_out.*net.LW'.*(-2*(-2)*aux./(1 + aux).^2);
    d_b_in = d_aux;
    
    % input layer weight update - IW
    N       = size(input,2);
    Nin     = size(input,1);
    Nhidden = size(net.IW,1);
    d_IW    = zeros(Nhidden,N,Nin);
    
    for i = 1:Nin
        d_IW(:,:,i) = d_aux.*input(i,:);
    end
    
    % Initialize the matrix J:
    J = zeros(N,(Nin + 2)*Nhidden + 1);
    
    % Jacobian for updating the parameters in the LM algorithm
    % -- with bias update:
    % - J = [weight_xj weight_yj bias_j_in weight_j bias_out]
%     J = [d_IW(:,:,1)' d_IW(:,:,2)' d_b_in' d_LW d_b_out*ones(N,1)]; 
    % -- without bias update:
    % - J = [weight_xj weight_yj weight_j]
    for i = 1:Nin
        J(:,Nhidden*(i-1)+1:i*Nhidden) = d_IW(:,:,i)';
    end
    J(:,i*Nhidden+1:(i+1)*Nhidden) = d_LW;
    J(:,(i+1)*Nhidden+1:(i+2)*Nhidden) = d_b_in';
    J(:,(i+2)*Nhidden+1:end) = d_b_out*ones(N,1);
    
    % Gradient:
    grad = J'*(Y - output.Y2)';
    
    % LM solution:  
    h_LM = -(J'*J + lambda*eye(size(J,2)))^-1*grad;
   
% --> RBF Network -> Ex. 3
elseif strcmp(net.name,'rbf')
    % Output layer weight update - LW 
    d_b_out = (-1); 
    d_LW = (d_b_out*output.Y1)';
    
    % Input layer weight and centers update - IW & center 
    N       = size(input,2);
    Nin     = size(input,1);
    Nhidden = size(net.centers,1);
    % d_V1    = zeros(Nhidden,N);
    % d_V1_c  = zeros(Nhidden,N);
    d_IW    = zeros(Nhidden,N,Nin);
    d_centers  = zeros(Nhidden,N,Nin);
    
    for i = 1:Nin
        % Input layer weight update - IW
        d_V1 = 2*net.IW(:,i).*(input(i,:).*ones(Nhidden,N)-net.centers(:,i)).^2;
        d_IW(:,:,i) = d_b_out*net.LW'.*(-output.Y1).*d_V1;
        % Center update - center
        d_V1_c = -2*(net.IW(:,i).^2*input(i,:)-net.IW(:,i).^2.*net.centers(:,i));
        d_centers(:,:,i) = d_b_out*net.LW'.*(-output.Y1).*d_V1_c;
    end
    
    % Initialize the matrix J:
    J = zeros(N,(Nin*2 + 1)*Nhidden);
    
    % Jacobian matrix:
    for i = 1:Nin
        J(:,Nhidden*(i-1)+1:i*Nhidden) = d_IW(:,:,i)';
        J(:,Nhidden*(i+Nin-1)+1:(i+Nin)*Nhidden) = d_centers(:,:,i)';
    end
    J(:,(i+Nin)*Nhidden+1:end) = d_LW;
    
    % Gradient:
    grad = J'*(Y - output.Y2)';
    
    % LM solution:  
    h_LM = - (J'*J + lambda*eye(size(J,2)))^-1*grad; % vector
    
else
    fprintf('<LMDerMat.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
end