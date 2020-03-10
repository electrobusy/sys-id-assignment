% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: kf_calc_Hx.m

function Hx = kf_calc_Hx(t, x, u)
    
    u = x(1);
    v = x(2);
    w = x(3);
    C_alpha_up = x(4);
    
    % Calculate Jacobian matrix of output dynamics
    Hx = zeros(3,4);
    
    u_w_2 = u^2 + w^2;
    Hx(1,1) = -w/u_w_2*(1 + C_alpha_up);
    % Hx(1,2) = 0;
    Hx(1,3) = u/u_w_2*(1 + C_alpha_up);
    Hx(1,4) = atan(w/u);
    
    Hx(2,1) = (v*u/u_w_2^(3/2))/(1 + (v/sqrt(u_w_2))^2);   
    Hx(2,2) = 1/sqrt(u_w_2)/(1 + (v/sqrt(u_w_2)));
    Hx(2,3) = (v*w/u_w_2^(3/2))/(1 + (v/sqrt(u_w_2))^2);
    % Hx(2,4) = 0;
    
    u_v_w_2 = u^2 + v^2 + w^2;
    Hx(3,1) = u/sqrt(u_v_w_2);
    Hx(3,2) = v/sqrt(u_v_w_2);
    Hx(3,3) = w/sqrt(u_v_w_2);
    % Hx(3,4) = 0;               
end
