function [update,grad] = BackPropDer(net,output,input,Y,lambda)

% --> FeedForward Network -> Ex. 4
if strcmp(net.name,'feedforward')
    % error - e
    e = (Y - output.Y2);
    
    % output bias update - b_out
    d_b_out = e*(-1); 
    
    % output layer weight update - LW
    d_LW = d_b_out*output.Y1';
    
    % input bias update - b_int
    aux = exp(-2*output.V1);
    d_aux = d_b_out*net.LW'.*(2*(-2)*aux./(1 + aux).^2);
    d_b_in = d_aux;
    
    % input layer weight update - IW
    Nin     = size(input,1);
    Nhidden = size(net.IW,1);
    d_IW    = zeros(Nhidden,Nin);
    for i = 1:Nin
        d_IW(:,i) = d_aux*input(i,:);
    end
    
    % Saving the updates
    update.d_LW = d_LW;
    update.d_IW = d_IW;
    update.d_b_out = d_b_out;
    update.d_b_in = d_b_in;
    
    % Gradient - for evaluation purposes during training:
    grad = [update.d_LW(:); update.d_IW(:)]; %update.d_b_out(:); update.d_b_in(:)];

% --> RBF Network -> Ex. 3
elseif strcmp(net.name,'rbf')
    % error - e
    e = (Y - output.Y2);
    
    % Output layer weight update - LW 
    d_b_out = (Y - output.Y2)*(-1); 
    d_LW = (d_b_out*output.Y1)';
    
    % Input layer weight and centers update - IW & center 
    Nin     = size(input,1);
    Nhidden = size(net.centers,1);
    d_V1 = zeros(Nhidden,Nin);
    d_V1_c = zeros(Nhidden,Nin);
    d_IW = zeros(Nhidden,Nin);
    d_centers = zeros(Nhidden,Nin);
    
    for i = 1:Nin
        % Input layer weight update - IW
        d_V1(:,i) = 2*net.IW(:,i).*(input(i)*ones(Nhidden,1)-net.centers(:,i)).^2;
        d_IW(:,i) = d_b_out*net.LW'.*(-output.Y1).*d_V1(:,i);
        % Center update - center
        d_V1_c(:,i) = -2*(net.IW(:,i).^2*input(i)-net.IW(:,i).^2.*net.centers(:,i));
        d_centers(:,i) = d_b_out*net.LW'.*(-output.Y1).*d_V1_c(:,i);
    end
    
    % Saving the updates
    update.d_LW = d_LW;
    update.d_IW = d_IW;
    update.d_centers = d_centers;
    
    % Gradient - for evaluation purposes during training:
    grad = [update.d_LW(:); update.d_IW(:) update.d_centers(:)];
else
    fprintf('<BackPropDer.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
end