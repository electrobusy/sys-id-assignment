function [update,grad] = BackPropDerMat(net,output,input,Y,lambda)

% --> FeedForward Network -> Ex. 4
if strcmp(net.name,'feedforward')
    % error - e
    e = (Y - output.Y2);
    
    % output bias update - b_out
    d_b_out = e*(-1); 
    
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
    d_IW    = zeros(Nhidden,Nin,N);
    
    for i = 1:Nin
        d_IW(:,i,:) = d_aux.*input(i,:);
    end
    
    % Saving the updates
    update.d_LW = sum(d_LW,1)/N;
    update.d_b_out = sum(d_b_out,2)/N;
    update.d_IW = sum(d_IW,3)/N;
    update.d_b_in = sum(d_b_in,2)/N;
    
    % Gradient - for evaluation purposes during training:
    grad = [update.d_LW(:); update.d_IW(:); update.d_b_out(:); update.d_b_in(:)];

% --> RBF Network -> Ex. 3
elseif strcmp(net.name,'rbf')
    % error - e
    e = (Y - output.Y2);
    
    % Output layer weight update - LW 
    d_b_out = e*(-1); 
    d_LW = (d_b_out.*output.Y1)';
    
    % Input layer weight and centers update - IW & center 
    N       = size(input,2);
    Nin     = size(input,1);
    Nhidden = size(net.centers,1);
    % d_V1    = zeros(Nhidden,N);
    % d_V1_c  = zeros(Nhidden,N);
    d_IW    = zeros(Nhidden,Nin,N);
    d_centers  = zeros(Nhidden,Nin,N);
    
    for i = 1:Nin
        % Input layer weight update - IW
        d_V1 = 2*net.IW(:,i).*(input(i,:).*ones(Nhidden,N)-net.centers(:,i)).^2;
        d_IW(:,i,:) = d_b_out.*net.LW'.*(-output.Y1).*d_V1;
        % Cen ter update - center
        d_V1_c = -2*(net.IW(:,i).^2*input(i,:)-net.IW(:,i).^2.*net.centers(:,i));
        d_centers(:,i,:) = d_b_out.*net.LW'.*(-output.Y1).*d_V1_c;
    end
    
    % Saving the updates
    update.d_LW = sum(d_LW,1)/N;
    update.d_IW = sum(d_IW,3)/N;
    update.d_centers = sum(d_centers,3)/N;
    
    % Gradient - for evaluation purposes during training:
    grad = [update.d_LW(:); update.d_IW(:); update.d_centers(:)];
else
    fprintf('<BackPropDerMat.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
end