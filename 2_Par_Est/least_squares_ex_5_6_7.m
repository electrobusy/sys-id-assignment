% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: least_squares_ex_5_6_7.m

clear all
close all;

%% Load the KF data:

load('reconstructed_data_KF.mat');

% % Data choice: 
% -- Extended Kalman Filter:
% Z_k_rec = Z_k_rec_ext;
% data_str = 'EKF';
%  -- Iterative Kalman Filter:
Z_k_rec = Z_k_rec_it;
data_str = 'IEKF';

Y = Cm; % We want to reconstruct get the envelope for Cm

max_order = 35; % to assess the model order performance

%% 5. Linear Regression: 
% p(x) = A*theta + epsillon 

M = 2; % Polynomial order -> (alpha + beta)^M
        % Cross terms should be taken into account too! -> C_alpha_beta,
        % C_alpha_beta^2, ... etc (depending in polynomial order)

% Define the regression matrix A with the reconstructed data:
A = reg_matrix(Z_k_rec(:,1),Z_k_rec(:,2),M);

%% -- Ordinary Least-Squares Estimator (OLS):
% % Apply the OLS algorithm: 
[ls_par_OLS,Y_OLS] = ord_least_squares(A,Y);

% % Plots:
plot_index = 1;
plot_Cm(Y,Y_OLS,Z_k_rec,plot_index,['OLS - ' data_str ' -> Reconstructed C_m for model order = ' num2str(M)]) % Y = Cm

%% -- Weighted Least-Squares Estimator (WLS):

% sigma = 0.112;
% R = sigma^2*eye(size(A,1));
% 
% % % Apply the OLS algorithm: 
% [ls_par_WLS,Y_WLS] = wei_least_squares(A,Y,R);
% 
% % % Plots:
% plot_index = 2;
% % % CHANGE ALL THE PLOTS
% plot_cm(Y_WLS,Z_k_rec,plot_index,'Data Reconstruction -> WLS') % Y = Cm

%% -- Recursive Least-Squares Estimator (RLS):

% P0 = 5000*eye(size(A,2)); % less confidence in the estimates
% lambda = 0.1; % forgeting factor
%  
% % % Apply the RLS algorithm: 
% [ls_par_RLS,Y_RLS] = rec_least_squares(A,Y,P0,ls_par_OLS,lambda);
% 
% % % Plots:
% plot_index = 3;
% plot_cm(Y_RLS,Z_k_rec,plot_index,'Data Reconstruction -> RLS') % Y = Cm

%% 6. Influence of the polynomial order in the accuracy of fit - for the OLS case

poly_order_vec = 1:max_order; % different polynomial orders

% Declaration and initialization of variables:
cost_OLS = zeros(1,length(poly_order_vec));

for i = poly_order_vec
    % % Define the regression matrix A with the reconstructed data:
    A = reg_matrix(Z_k_rec(:,1),Z_k_rec(:,2),i);
    
    % % Apply OLS:
    [ls_par_OLS,Y_OLS] = ord_least_squares(A,Y);

    % % Use quadratic cost function as a metric for the accuracy fit:
    cost_OLS(i) = (1/2)*sum((Y - Y_OLS).^2);
end

plot_index = 4;
figure(plot_index);
% % Find the order that yields the minimum cost value:
[~,min_order] =  min(cost_OLS);
% % Plot the Cost x Polynomial Order
plot(poly_order_vec,cost_OLS);
yL = get(gca,'YLim'); 
p1 = line([min_order min_order],yL,'Linestyle','-.','Color','r',...
    'Linewidth',1); % horizontal line of the order that yields the best polynomial fit
xL = get(gca,'XLim'); 
p2 = line(xL,[cost_OLS(min_order) cost_OLS(min_order)],'Linestyle','-.','Color','r',...
    'Linewidth',1); % horizontal line of the order that yields the best polynomial fit
legend('cost', ['Best poly. order: ' num2str(min_order)],['Minimum cost: ' num2str(cost_OLS(min_order))]);
xlabel('Polynomial order [-]');
ylabel('Cost J [-]');
axis([])


%% 7. Model validation: perform both statistical and model-error based validation:
% Now that the best order for the polynomial was found, the goal now is to
% perform a statistical analysis. 

% % Define the regression matrix A with the reconstructed data:
A = reg_matrix(Z_k_rec(:,1),Z_k_rec(:,2),min_order);

% % Apply OLS:
[ls_par_OLS,Y_OLS] = ord_least_squares(A,Y);

% % Statistical validation (Analysis of parameter covariances):
plot_index = 5;
figure(plot_index);
bar(ls_par_OLS); % Estimated coefficients 
xlabel(['Parameter (C_{m_{\alpha^n\beta^m}} = \theta_i, i = 1,...,' num2str(length(ls_par_OLS)) ') [-]']);
ylabel('Parameter value');
COV_mat = (A'*A)^-1;
plot_index = 6;
figure(plot_index);
bar(diag(COV_mat)); % Parameter variances
xlabel(['Parameter (C_{m_{\alpha^n\beta^m}} = \theta_i, i = 1,...,' num2str(length(ls_par_OLS)) ') [-]']);
ylabel('Variance');
[min_var,idx] = min(COV_mat(:));
[idx_min_row, idx_min_col] = ind2sub(size(COV_mat),idx);
fprintf('Parameters that are correlated the most: Cov{ theta_%d, theta_%d } = %f\n',idx_min_row,idx_min_col,min_var);

% % Model-error validation (Model residual analysis):
n_lags = 300;
conf_up_bound = 1.96/sqrt(length(Y));
res_error = Y_OLS - Y;
lambda = xcorr(res_error,res_error,n_lags);

plot_index = 7;
figure(plot_index);
plot(-n_lags:1:n_lags,lambda,-n_lags:1:n_lags,conf_up_bound*ones(1,length(lambda)),-n_lags:1:n_lags,-conf_up_bound*ones(1,length(lambda)));
xlabel('Number of lags');
ylabel('Autocorrelation [-]');
legend('Autocorrelation','95% confidence upper bound','95% confidence lower bound');