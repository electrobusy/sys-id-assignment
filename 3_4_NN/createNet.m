% For purposes of doing the data reconstruction of the F-16 aircraft, we
% have:
% - IN = 2 -> we have two entries: alpha_t and beta_t
% - ON = 1 -> we want to obtain one output, in this case C_m
% Thus, the only parameter that we vary is HN, the number of neurons in the
% hidden layer!
function net = createNet(type,IN,HN,ON,mu,N_epochs)

% Feedfoward Network
if strcmp(type,'feedforward')
    % -> Variables Fields:
    r = 2;
    % r = 1/sqrt(HN);
    % r = sqrt(6/(IN+ON));
    % r = 4*sqrt(6/(IN+ON));
    % r = sqrt(1/(IN+ON));
    x = r;
    net.IW = -x + (x-(-x)).*rand(HN,IN);    % initialize the network from a Uni. distribution - [-x,x]        
    y = r;
    net.LW = -y + (y-(-y)).*rand(ON,HN);    % initialize the network from a Uni. distribution - [-y,y]
    net.b = {rands(HN,1);rands(ON,1)}; 
    net.range = zeros(IN,2);        % range of the input data [won't be used]
    net.trainParam.epochs = N_epochs;
    net.trainParam.goal = 1e-10;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.mu = mu;
    net.trainParam.mu_dec = 0.1;
    net.trainParam.mu_inc = 10;
    net.trainParam.mu_max = 1e+8;
    % -> [Fixed] Fields:
    net.name = {'feedforward'};
    net.trainFunct = {'tansig';'purelin'};
    net.trainAlg = {'trainlm'};
% RBF Network
elseif strcmp(type,'rbf')
    z = 0.4;
    net.centers = -z + (z-(-z)).*rand(HN,IN);      % Later one, a choice is made by using clustering techniques
    net.trainAlg = {'trainlm'};     % [Fixed]
    net.trainFunct = {'radbas'};    % [Fixed]
    net.range = zeros(IN,2);        % range of the input data [won't be used]
    net.N_centers = [HN IN];
    net.name = {'rbf'};             % [Fixed]
    net.trainParam.epochs = N_epochs;
    net.trainParam.goal = 1e-6;
    net.trainParam.min_grad = 1e-10;
    net.trainParam.mu = mu;
    net.trainParam.mu_dec = 0.1;
    net.trainParam.mu_inc = 10;
    net.trainParam.mu_max = 1e+10; 
    x = 2;
    net.IW = -0 + (x-(-0)).*rand(HN,IN); % initialize the network from a Uni. distribution - [-x,x]
    y = 2;
    net.LW = -y + (y-(-y)).*rand(ON,HN); % initialize the network from a Uni. distribution - [-y,y]
else
    fprintf('<createNet.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
end

end
