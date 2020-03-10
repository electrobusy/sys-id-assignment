% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: kf_calc_h.m

function zpred = kf_calc_h(t, x, u)
    u = x(1);
    v = x(2);
    w = x(3);
    C_alpha_up = x(4);
    
    alpha_true = atan(w/u);
    beta_true = atan(v/sqrt(u^2 + w^2));
    V_true = sqrt(u^2 + v^2 + w^2);
    
    alpha_m = alpha_true*(1 + C_alpha_up); 
    beta_m = beta_true;
    V_m = V_true;
    
    zpred = [alpha_m beta_m V_m]';
end
    