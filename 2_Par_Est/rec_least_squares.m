% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: rec_least_squares.m

% Computes the least-squares with forgeting factor 
function [ls_par,Y_Val] = rec_least_squares(A,Y,P0,c0,lambda)

P_kk = zeros(size(A,2),size(A,2),size(A,1));
c_kk = zeros(size(A,2),size(A,1));

% Initialization for the Recursive LS
P_kp1 = P0;
c_kp1 = c0;

for i = 1:size(A,1)
    
    y_kp1  = Y(i);
    
    P_k = P_kp1;    
    c_k = c_kp1;
    
    % 3) Formulation of the new regression vector/matrix for new data
    % points:
    a_kp1 = A(i,:); 
    
    % 4) Calculate Kalman gain 
    % --> Precalculate the P_k * a_kp1' matrix for speed (6x improvement)
    a_kp1P_k = P_k * a_kp1';
    % --> Computation of the gain
    K_kp1 = a_kp1P_k / (a_kp1 * a_kp1P_k + lambda);
    
    % 5) Update parameters
    delta_c =  K_kp1 * (y_kp1 - a_kp1 * c_k);
    c_kp1 = c_k + delta_c;
%    c_kp1 = c_k + P_k*a_kp1'* (y_kp1 - a_kp1 * c_k) + K_kp1*a_kp1*P_k*a_kp1'*(y_kp1 - a_kp1 * c_k);
    
    % 6) Update covariance matrix
    P_kupdate = K_kp1 * a_kp1 * P_k;
    P_kp1 = P_k - P_kupdate;      
    
    % Save the values: 
    P_kk(:,:,i) = P_kp1;
    c_kk(:,i) = c_kp1;
end

% We want the last estimate, because that is the best one:
ls_par = c_kp1; 
Y_Val = A * ls_par;

end