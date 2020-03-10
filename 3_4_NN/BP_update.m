% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: BP_update.m

function [net] = BP_update(net,update)
    % --> FeedForward Network -> Ex. 4
    if strcmp(net.name,'feedforward')      
        net.IW = net.IW - net.trainParam.mu*update.d_IW; % inner weights
        net.LW = net.LW - net.trainParam.mu*update.d_LW; % outer weights
        net.b{2} = net.b{2} - net.trainParam.mu*update.d_b_out; % output bias
        net.b{1} = net.b{1} - net.trainParam.mu*update.d_b_in; % inner bias
        
    % --> RBF Network -> Ex. 3
    elseif strcmp(net.name,'rbf')
        net.IW = net.IW - net.trainParam.mu*update.d_IW; % inner weights
        net.LW = net.LW - net.trainParam.mu*update.d_LW; % outer weights
        net.centers = net.centers - net.trainParam.mu*update.d_centers; % RBF centers
    else 
        fprintf('<BP_update.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
    end
end