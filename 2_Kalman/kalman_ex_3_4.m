% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: kalman_ex_3_4.m

addpath(genpath('Extra Functions'));

close all
clear all

% Loading data
load_f16data2018

% --------------
% Set simulation parameters 
% --------------
dt              = 0.01; % time-step
N               = length(Z_k(:,1)); % Number of elements for processing the data

% For the Iterated Extended Kalman Filter
epsilon         = 0.0001;
doIEKF          = 1;
maxIterations   = 100;

% For the plots
printfigs = 0;
figpath = '';

% --------------
% Set initial values for states and statistics (for both Kalman Filters)
% --------------
Ex_0    = [1 0 0 0]'; % initial estimate of optimal value of x_k_1k_1 (TUNE THIS VALUE LATER)
% x_0     = [0 0 0 0]'; % initial state -> Let us start the model from zero
% we don't have this, since we are dealing with real data.
n       = 4; % number of states -> x_k = [u v w C_alpha_up]
nm      = 3; % number of measurements -> z_k = [alpha_m beta_m V_m]
m       = 3; % number of inputs -> u_k = [u_dot v_dot w_dot]

% Initial estimate for covariance matrix
stdx_0  = 100; % tunning parameter for the covariance matrix
P_0     = stdx_0^2*eye(4);

% Dynamics equation (linear) - Info: 
F = zeros(4); 
B = [1 0 0;
     0 1 0;
     0 0 1;
     0 0 0]; % input matrix

% System/Process noise statistics:
q = 1e-3;
Q = diag([q^2 q^2 q^2 0]);

% Measurement noise statistics:
R = diag([0.01^2 0.0058^2 0.112^2]);

% Variables - initialization for the Kalman Filter application: 
% --> Extended Kalman Filter:
XX_k1k1_ext = zeros(n, N);
ZZ_k1k1_ext = zeros(nm, N); % Estimated Output
PP_k1k1_ext = zeros(n,n, N);
STDx_cor_ext = zeros(n, N);
ERROR_cor_ext = zeros(n,N);
z_pred_ext = zeros(nm,N);
ZZ_k1k1_ext_alpha_t = zeros(1, N);

x_k_1k_1_ext = Ex_0; % x(0|0)=E{x_0}
P_k_1k_1_ext = P_0; % P(0|0)=P(0)

% --> Iterated Extended Kalman Filter:
XX_k1k1_it = zeros(n, N);
ZZ_k1k1_it = zeros(nm, N); % Estimated Output
PP_k1k1_it = zeros(n,n, N);
STDx_cor_it = zeros(n, N);
ERROR_cor_it = zeros(n,N);
z_pred_it = zeros(nm, N);
ZZ_k1k1_it_alpha_t = zeros(1, N);

IEKFitcount = zeros(N, 1);

x_k_1k_1_it = Ex_0; % x(0|0)=E{x_0}
P_k_1k_1_it = P_0; % P(0|0)=P(0)

% --------------
% Run the Extended and Iterative Kalman filter
% --------------

% Important for the integration part in the 1rst step:
ti = 0; 
tf = dt;

% Run the filter through all N samples
for k = 1:N
    % 1) Prediction x(k+1|k) --> without noise
    % @kf_calc_f --> when we want to pass the function inside the rk4
    % function, so we can change it inputs inside that function
    % --> Extended Kalman Filter:
    [~, x_kk_1_ext] = rk4(@kf_calc_f, F, B, x_k_1k_1_ext, U_k(k,:)', [ti tf]); 
    % --> Iterated Extended Kalman Filter:
    [~, x_kk_1_it] = rk4(@kf_calc_f, F, B, x_k_1k_1_it, U_k(k,:)', [ti tf]); 
    
    % 2) z(k+1|k) (predicted output) --> without noise
    % --> Extended Kalman Filter:
    z_kk_1_ext = kf_calc_h(0, x_kk_1_ext, U_k(k,:)'); %x_kk_1.^3; 
    z_pred_ext(:,k) = z_kk_1_ext;
    % --> Iterated Extended Kalman Filter:
    z_kk_1_it = kf_calc_h(0, x_kk_1_it, U_k(k,:)'); %x_kk_1.^3; 
    z_pred_it(:,k) = z_kk_1_it;

    % 3) Calc Phi(k+1,k) and Gamma(k+1, k) (equal for both Filters - because 
    % the Dynamics is linear)
    Fx = F; % perturbation of f(x,u,t) --> since the Dynamics is linear, 
            % it is the same matrix F
    G = eye(4); % noise input matrix --> all the states are influenced by noise
    % The continuous to discrete time transformation of Df(x,u,t)
    [~, Psi] = c2d(Fx, B, dt); 
    [Phi, Gamma] = c2d(Fx, G, dt);   
    
    % 4) P(k+1|k) (prediction covariance matrix)
    % --> Extended Kalman Filter:
    P_kk_1_ext = Phi*P_k_1k_1_ext*Phi' + Gamma*Q*Gamma'; 
    P_pred_ext = diag(P_kk_1_ext);
    stdx_pred_ext = sqrt(P_pred_ext); % standard deviation of the prediction 
                                      % state error - plotting purposes
    % --> Iterated Extended Kalman Filter:
    P_kk_1_it = Phi*P_k_1k_1_it*Phi' + Gamma*Q*Gamma'; 
    P_pred_it = diag(P_kk_1_it);
    stdx_pred_it = sqrt(P_pred_it); % standard deviation of the prediction 
                                    % state error - plotting purposes
    
    % ----- Extended (Correction and Gain)
    % 6) Correction
    Hx_ext = kf_calc_Hx(0, x_kk_1_ext, U_k(k,:)'); % perturbation of h(x,u,t)
    % Pz(k+1|k) (covariance matrix of innovation)
    Ve = (Hx_ext*P_kk_1_ext * Hx_ext' + R); 

    % 7) K(k+1) (gain)
    K_ext = P_kk_1_ext * Hx_ext' / Ve;
    % Calculate optimal state x(k+1|k+1) 
    x_k_1k_1_ext = x_kk_1_ext + K_ext * (Z_k(k,:)' - z_kk_1_ext); % new estimate
    % Determine the state error
    state_error_ext =  x_k_1k_1_ext - x_kk_1_ext; % determine de approx. state error
    
    % ----- Iterative (Correction, Gain and iterative step)
    % do the iterative part
    eta2    = x_kk_1_it;
    err     = 2*epsilon;

    itts    = 0;
    while (err > epsilon)
        if (itts >= maxIterations)
            fprintf('Terminating IEKF: exceeded max iterations (%d)\n', maxIterations);
            break;
        end
        itts    = itts + 1;
        eta1    = eta2;

        % Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
        Hx_it   = kf_calc_Hx(0, eta1, U_k(k,:)'); 
            
        % Check observability of state
        if (k == 1 && itts == 1)
            rankHF = kf_calcObsRank(Hx_it, Fx);
            if (rankHF < n)
                warning('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);
            end
        end
            
        % The innovation matrix
        Ve  = (Hx_it*P_kk_1_it*Hx_it' + R);

        % calculate the Kalman gain matrix
        K_it       = P_kk_1_it * Hx_it' / Ve;
        % new observation state
        z_p     = kf_calc_h(0, eta1, U_k(k,:)') ;%fpr_calcYm(eta1, u);

        eta2    = x_kk_1_it + K_it * (Z_k(k,:)' - z_p - Hx_it*(x_kk_1_it - eta1));
        err     = norm((eta2 - eta1), inf) / norm(eta1, inf);
    end
    IEKFitcount(k)    = itts;
    x_k_1k_1_it       = eta2;
    state_error_it    = x_k_1k_1_it - x_kk_1_it;

    % 8) P(k+1|k+1) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1; 
    % --> Extended Kalman Filter:
    P_k_1k_1_ext = (eye(n) - K_ext*Hx_ext) * P_kk_1_ext;  
    P_cor_ext = diag(P_k_1k_1_ext);
    stdx_cor_ext = sqrt(P_cor_ext); % standard deviation of the estimation 
    
    % --> Iterated Extended Kalman Filter:
    P_k_1k_1_it = (eye(n) - K_it*Hx_it) * P_kk_1_it * (eye(n) - K_it*Hx_it)' + K_it*R*K_it';  
    P_cor = diag(P_k_1k_1_it);
    stdx_cor_it = sqrt(P_cor);
    
    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % Store results
    % --> Extended Kalman Filter:
    XX_k1k1_ext(:,k) = x_k_1k_1_ext;
    ZZ_k1k1_ext(:,k) = kf_calc_h(0, x_k_1k_1_ext, U_k(k,:)');
    ZZ_k1k1_ext_alpha_t(k) = atan(x_k_1k_1_ext(3)/x_k_1k_1_ext(1))/(1 + x_k_1k_1_ext(4)); % Remove the bias, to get alpha_true!
    PP_k1k1_ext(:,:,k) = P_k_1k_1_ext;
    STDx_cor_ext(:,k) = stdx_cor_ext;
    ERROR_cor_ext(:,k) = state_error_ext;
    % --> Iterated Extended Kalman Filter:
    XX_k1k1_it(:,k) = x_k_1k_1_it;
    ZZ_k1k1_it(:,k) = kf_calc_h(0, x_k_1k_1_it, U_k(k,:)');
    ZZ_k1k1_it_alpha_t(k) = atan(x_k_1k_1_it(3)/x_k_1k_1_it(1))/(1 + x_k_1k_1_it(4)); % Remove the bias, to get alpha_true!
    PP_k1k1_it(:,:,k) = P_k_1k_1_it;
    STDx_cor_it(:,k) = stdx_cor_it;
    ERROR_cor_it(:,k) = state_error_it;
end

%% Plots - EKF and IEKF together

% Time axis:
t = 0:dt:dt*(N-1);

% % TWO FILTERS:
% Plot - Extended and Iterative Extended KF -> States
figure(printfigs+1);
suptitle('Estimated States');

subplot(4,1,1);
plot(t,XX_k1k1_ext(1,:),t,XX_k1k1_it(1,:));
xlabel('t [s]');
ylabel('u [m/s]');
legend('EKF', 'IEKF');

subplot(4,1,2);
plot(t,XX_k1k1_ext(2,:),t,XX_k1k1_it(2,:));
xlabel('t [s]');
ylabel('v [m/s]');
legend('EKF', 'IEKF');

subplot(4,1,3);
plot(t,XX_k1k1_ext(3,:),t,XX_k1k1_it(3,:));
xlabel('t [s]');
ylabel('w [m/s]');
legend('EKF', 'IEKF');

subplot(4,1,4);
plot(t,XX_k1k1_ext(4,:),t,XX_k1k1_it(4,:));
xlabel('t [s]');
ylabel('C_{\alpha_{up}} [-]');
legend('EKF', 'IEKF');

% Plot - Extended KF - Outputs
figure(printfigs+2);
suptitle('Outputs: Real x Estimated - EKF');

subplot(3,1,1);
plot(t,Z_k(:,1),t,ZZ_k1k1_ext(1,:),t,ZZ_k1k1_ext_alpha_t);
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('Real','EKF - bias','EKF - no bias');

subplot(3,1,2);
plot(t,Z_k(:,2),t,ZZ_k1k1_ext(2,:));
xlabel('t [s]');
ylabel('\beta [rad]');
legend('Real','EKF');

subplot(3,1,3);
plot(t,Z_k(:,3),t,ZZ_k1k1_ext(3,:));
xlabel('t [s]');
ylabel('V_t [m/s]');
legend('Real','EKF');

% Plot - Iterative Extended KF - Outputs
figure(printfigs+3);
suptitle('Outputs: Real x Estimated - IEKF');

subplot(3,1,1);
plot(t,Z_k(:,1),t,ZZ_k1k1_it(1,:),t,ZZ_k1k1_it_alpha_t);
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('Real','IEKF - bias','IEKF - no bias');

subplot(3,1,2);
plot(t,Z_k(:,2),t,ZZ_k1k1_it(2,:));
xlabel('t [s]');
ylabel('\beta [rad]');
legend('Real','IEKF');

subplot(3,1,3);
plot(t,Z_k(:,3),t,ZZ_k1k1_it(3,:));
xlabel('t [s]');
ylabel('V_t [m/s]');
legend('Real','IEKF');

%% Plots - Separated figures:
% % STATES and OUTPUTS:
% --- Extended Kalman Filter:

% - States
% Longitudinal velocity u
figure(printfigs+4);
plot(t,XX_k1k1_ext(1,:));
xlabel('t [s]');
ylabel('u [m/s]');
title('EKF: State u');

% Lateral velocity v
figure(printfigs+5);
plot(t,XX_k1k1_ext(2,:));
xlabel('t [s]');
ylabel('v [m/s]');
title('EKF: State v');

% Vertical velocity w
figure(printfigs+6);
plot(t,XX_k1k1_ext(3,:));
xlabel('t [s]');
ylabel('w [m/s]');
title('EKF: State w');

% Bias C_n_alpha_up 
figure(printfigs+7);
plot(t,XX_k1k1_ext(4,:));
xlabel('t [s]');
ylabel('C_{\alpha_{up}} [-]');
xL = get(gca,'XLim'); 
p1= line(xL,[XX_k1k1_ext(4,end) XX_k1k1_ext(4,end)],'Linestyle','--','Color','r',...
    'Linewidth',1);
legend('C_{\alpha_{up}}',['C_{\alpha_{up}} steady-state: ' num2str(XX_k1k1_ext(4,end))]);
title('EKF: State C_{\alpha_{up}}');

% - Output
% Alpha
figure(printfigs+8);
plot(t,Z_k(:,1),t,ZZ_k1k1_ext(1,:));
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('Real','EKF');
title('EKF: Output \alpha');

% Beta
figure(printfigs+9);
plot(t,Z_k(:,2),t,ZZ_k1k1_ext(2,:));
xlabel('t [s]');
ylabel('\beta [rad]');
legend('Real','EKF');
title('EKF: Output \beta');

% V_t
figure(printfigs+10);
plot(t,Z_k(:,3),t,ZZ_k1k1_ext(3,:));
xlabel('t [s]');
ylabel('V_t [m/s]');
legend('Real','EKF');
title('EKF: Output V_t');

% --- Iterative Extended Kalman Filter:

% - States
% Longitudinal velocity u
figure(printfigs+11);
plot(t,XX_k1k1_it(1,:));
xlabel('t [s]');
ylabel('u [m/s]');
title('IEKF: State u');

% Lateral velocity v
figure(printfigs+12);
plot(t,XX_k1k1_it(2,:));
xlabel('t [s]');
ylabel('v [m/s]');
title('IEKF: State v');

% Vertical velocity w
figure(printfigs+13);
plot(t,XX_k1k1_it(3,:));
xlabel('t [s]');
ylabel('w [m/s]');
title('IEKF: State w');

% Bias C_n_alpha_up 
figure(printfigs+14);
plot(t,XX_k1k1_it(4,:));
xlabel('t [s]');
ylabel('C_{\alpha_{up}} [-]');
xL = get(gca,'XLim'); 
p1= line(xL,[XX_k1k1_it(4,end) XX_k1k1_it(4,end)],'Linestyle','--','Color','r',...
    'Linewidth',1);
legend('C_{\alpha_{up}}',['C_{\alpha_{up}} steady-state: ' num2str(XX_k1k1_it(4,end))]);
title('IEKF: State C_{\alpha_{up}}');

% - Output
% Alpha
figure(printfigs+15);
plot(t,Z_k(:,1),t,ZZ_k1k1_it(1,:));
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('Real','IEKF');
title('IEKF: Output \alpha');

% Beta
figure(printfigs+16);
plot(t,Z_k(:,2),t,ZZ_k1k1_it(2,:));
xlabel('t [s]');
ylabel('\beta [rad]');
legend('Real','IEKF');
title('IEKF: Output \beta');

% V_t
figure(printfigs+17);
plot(t,Z_k(:,3),t,ZZ_k1k1_it(3,:));
xlabel('t [s]');
ylabel('V_t [m/s]');
legend('Real','IEKF');
title('EKF: Output V_t');

%% Analysis of the state estimation error

% --- Extended Kalman Fitler:
% Longitudinal velocity u
figure(printfigs+18);
plot(t,ERROR_cor_ext(1,:),t,STDx_cor_ext(1,:),t,-STDx_cor_ext(1,:));
xlabel('t [s]');
ylabel('[m/s]');
legend('\epsilon_u','\sigma_{\epsilon_u}','-\sigma_{\epsilon_u}');
axis([0 100 -0.01 0.01]);
title('State u estimation error - EKF');

% Lateral velocity v
figure(printfigs+19);
plot(t,ERROR_cor_ext(2,:),t,STDx_cor_ext(2,:),t,-STDx_cor_ext(2,:));
xlabel('t [s]');
ylabel('[m/s]');
legend('\epsilon_v','\sigma_{\epsilon_v}','-\sigma_{\epsilon_v}');
axis([0 100 -0.01 0.01]);
title('State v estimation error - EKF');

% Vertical velocity w
figure(printfigs+20);
plot(t,ERROR_cor_ext(3,:),t,STDx_cor_ext(3,:),t,-STDx_cor_ext(3,:));
xlabel('t [s]');
ylabel('[m/s]')
legend('\epsilon_w','\sigma_{\epsilon_w}','-\sigma_{\epsilon_w}');
axis([0 100 -0.01 0.01]);
title('State w estimation error - EKF');

% Bias C_n_alpha_up
figure(printfigs+21);
plot(t,ERROR_cor_ext(4,:),t,STDx_cor_ext(4,:),t,-STDx_cor_ext(4,:));
xlabel('t [s]');
ylabel('[-]')
legend('\epsilon_{C_{\alpha_{up}}}','\sigma_{\epsilon_{C_{\alpha_{up}}}}','-\sigma_{\epsilon_{C_{\alpha_{up}}}}');
axis([0 100 -0.01 0.01]);
title('State C_{\alpha_{up}} estimation error - EKF');

% --- Iterative Kalman Filter:
% Longitudinal velocity u
figure(printfigs+22);
plot(t,ERROR_cor_it(1,:),t,STDx_cor_it(1,:),t,-STDx_cor_it(1,:));
xlabel('t [s]');
ylabel('[m/s]')
legend('\epsilon_u','\sigma_{\epsilon_u}','-\sigma_{\epsilon_u}');
axis([0 100 -0.01 0.01]);
title('State u estimation error - IEKF');

% Lateral velocity v
figure(printfigs+23);
plot(t,ERROR_cor_it(2,:),t,STDx_cor_it(2,:),t,-STDx_cor_it(2,:));
xlabel('t [s]');
ylabel('[m/s]')
legend('\epsilon_v','\sigma_{\epsilon_v}','-\sigma_{\epsilon_v}');
axis([0 100 -0.01 0.01]);
title('State v estimation error - IEKF');

% Vertical velocity w
figure(printfigs+24);
plot(t,ERROR_cor_it(3,:),t,STDx_cor_it(3,:),t,-STDx_cor_it(3,:));
xlabel('t [s]');
ylabel('[m/s]')
legend('\epsilon_w','\sigma_{\epsilon_w}','-\sigma_{\epsilon_w}');
axis([0 100 -0.01 0.01]);
title('State w estimation error - IEKF');

% Bias C_n_alpha_up
figure(printfigs+25);
plot(t,ERROR_cor_it(4,:),t,STDx_cor_it(4,:),t,-STDx_cor_it(4,:));
xlabel('t [s]');
ylabel('[-]');
legend('\epsilon_{C_{\alpha_{up}}}','\sigma_{\epsilon_{C_{\alpha_{up}}}}','-\sigma_{\epsilon_{C_{\alpha_{up}}}}');
axis([0 100 -0.01 0.01]);
title('State C_{\alpha_{up}} estimation error - IEKF');

%% Observability analysis: 

% Define simbolically the states:
syms('u', 'v', 'w', 'C_n_alpha_up');
% Define simbolically the inputs:
syms('u_dot', 'v_dot', 'w_dot');

% Define state vector: 
x_syms = [u; v; w; C_n_alpha_up];
% Define input vector:
u_syms = [u_dot; v_dot; w_dot];

% Define state transition function
faug = [u_dot; v_dot; w_dot; 0];

% Define state observation function
haug = [
       atan(w/u)*(1 + C_n_alpha_up);
       atan(v/sqrt(u^2 + w^2));
       sqrt(u^2 + v^2 + w^2)
       ];
   
% Use the kf_calcNonlinObsRank function to calculate the rank of the observation matrix
[rankObsaug,Obs_mat] = kf_calcNonlinObsRank(faug, haug, x_syms, Ex_0);
%disp(Obs_mat)
if (rankObsaug >= n)
    fprintf('Augmented Observability matrix is of Full Rank: the augmented state is Observable!\n');
else
    fprintf('Augmented Observability matrix is of NOT Full Rank: the augmented state is NOT Observable!\n');
end

%% Alpha reconstruction plots:

figure(printfigs+27);
plot(t,Z_k(:,1),t,ZZ_k1k1_ext(1,:),t,ZZ_k1k1_ext_alpha_t);
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('\alpha_m','\alpha','\alpha_{true}');
title('EKF: \alpha reconstruction');

figure(printfigs+28);
plot(t,Z_k(:,1),t,ZZ_k1k1_it(1,:),t,ZZ_k1k1_it_alpha_t);
xlabel('t [s]');
ylabel('\alpha [rad]');
legend('\alpha_m','\alpha','\alpha_{true}');
title('IEKF: \alpha reconstruction');

%% Save reconstructed data:
Z_k_rec_ext = zeros(N, nm);
Z_k_rec_it = zeros(N, nm);

% --> Extended KF:
Z_k_rec_ext(:,1) = ZZ_k1k1_ext_alpha_t; 
Z_k_rec_ext(:,2) = ZZ_k1k1_ext(2,:);
Z_k_rec_ext(:,3) = ZZ_k1k1_ext(3,:);

% --> Iterative Extended KF:
Z_k_rec_it(:,1) = ZZ_k1k1_it_alpha_t; 
Z_k_rec_it(:,2) = ZZ_k1k1_it(2,:);
Z_k_rec_it(:,3) = ZZ_k1k1_it(3,:);

% Now we save the data in:
% 1) Parameter Estimator folder
savefile = '../2_Par_Est/reconstructed_data_KF';
save(savefile,'Cm','Z_k_rec_ext','Z_k_rec_it');

% 2) Neural Network folder
savefile = '../3_4_NN/reconstructed_data_KF';
save(savefile,'Cm','Z_k_rec_ext','Z_k_rec_it');