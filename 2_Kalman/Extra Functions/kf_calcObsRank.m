% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: kf_calcObsRank.m

function r = kf_calcObsRank(H, Fx)

    nstates = size(Fx,1); % number of states

    F = eye(size(Fx));  
    Rank = [];
    for i = 1:(nstates-1)
       Rank = [ Rank; H*F ];
       F = F*Fx;
    end
    Rank = [ Rank; H*F ];
    r    = rank(Rank);
end

