% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: kf_calc_f.m

function xdot = kf_calc_f(t,x,u,F,B)
    xdot = F*x+B*u;
end
