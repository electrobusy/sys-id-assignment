% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: LM_update.m

function [net] = LM_update(net,h_LM)
    % Important parameters:
    Nin = size(net.IW,2);
    Nhidden = size(net.IW,1);
    
    % --> FeedForward Network -> Ex. 4
    if strcmp(net.name,'feedforward')      
        % Previous paramaters in a matrix:
        % -- with bias update:
          p = [net.IW(:); net.LW'; net.b{1,1}; net.b{2,1}];
        % -- without bias update:
%         p = [net.IW(:); net.LW'];
    
        % Update parameters:
        p = p + h_LM;
    
        % Recover new parameters:
        % -- with bias update:
      for i = 1:Nin
          net.IW(:,i) = p(Nhidden*(i-1)+1:i*Nhidden);
      end
      net.LW = p(i*Nhidden+1:(i+1)*Nhidden)';
      net.b{1,1} = p((i+1)*Nhidden+1:(i+2)*Nhidden);
      net.b{2,1} = p((i+2)*Nhidden+1);
%         % -- without bias update:
%         for i = 1:Nin
%             net.IW(:,i) = p(Nhidden*(i-1)+1:i*Nhidden);
%         end
%         net.LW = p(i*Nhidden+1:end)';
        
    % --> RBF Network -> Ex. 3
    elseif strcmp(net.name,'rbf')
        % Previous paramaters in a matrix:
        p = [net.IW(:); net.centers(:); net.LW'];
    
        % Update parameters:
        p = p + h_LM;
    
        % Recover new parameters:
        for i = 1:Nin
            net.IW(:,i) = p(Nhidden*(i-1)+1:i*Nhidden);
            net.centers(:,i) = p(Nhidden*(i+Nin-1)+1:(i+Nin)*Nhidden);
        end
        net.LW = p((i+Nin)*Nhidden+1:end)';
    else 
        fprintf('<LM_update.m> Supplied network type is not correct. Must be a feedforward or rbf network ... \n');
    end
end