% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: ExtendedKalmanFilter.m

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the Extended Kalman filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% Important for the integration part in the first step:
ti = 0;  
tf = dt;

n = length(x_k); % n: state dimension

% Run the filter through all N samples
for k = 1:N
    % Prediction x(k+1|k) 
    [t, x_kk_1] = rk4(@kf_calc_f, x_k_1k_1, U_k(k), [ti tf]); 

    % z(k+1|k) (predicted output)
    z_kk_1 = kf_calc_h(0, x_kk_1, U_k(k)); %x_kk_1.^3; 
    z_pred(k) = z_kk_1;

    % Calc Phi(k+1,k) and Gamma(k+1, k)
    Fx = kf_calc_Fx(0, x_kk_1, U_k(k)); % perturbation of f(x,u,t)
    G = [1]; % noise input matrix
    % the continuous to discrete time transformation of Df(x,u,t)
    [dummy, Psi] = c2d(Fx, B, dt);   
    [Phi, Gamma] = c2d(Fx, G, dt);   
    
    % P(k+1|k) (prediction covariance matrix)
    P_kk_1 = Phi*P_k_1k_1*Phi' + Gamma*Q*Gamma'; 
    P_pred = diag(P_kk_1);
    stdx_pred = sqrt(diag(P_kk_1));

    % Correction
    Hx = kf_calc_Hx(0, x_kk_1, U_k(:,k)); % perturbation of h(x,u,t)
    % Pz(k+1|k) (covariance matrix of innovation)
    Ve = (Hx*P_kk_1 * Hx' + R); 

    % K(k+1) (gain)
    K = P_kk_1 * Hx' / Ve;
    % Calculate optimal state x(k+1|k+1) 
    x_k_1k_1 = x_kk_1 + K * (Z_k(:,k) - z_kk_1); 

    % P(k+1|k+1) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1; 
    P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 * (eye(n) - K*Hx)' + K*R*K';  
    
    P_cor = diag(P_k_1k_1);
    stdx_cor = sqrt(diag(P_k_1k_1));

    % Next step
    ti = tf; 
    tf = tf + dt;
    
    % store results
    XX_k1k1(k) = x_k_1k_1;
    PP_k1k1(k) = P_k_1k_1;
    STDx_cor(k) = stdx_cor;
end

time2 = toc;

% calculate state estimation error (in real life this is unknown!)
EstErr = XX_k1k1-X_k;

fprintf('EKF state estimation error RMS = %d, completed run with %d samples in %2.2f seconds.\n', sqrt(mse(EstErr)), N, time2);


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

plotID = 1000;
figure(plotID);
set(plotID, 'Position', [1 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(Z_k, 'k');
title('True state (blue) and Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateMeasurement');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1001;
figure(plotID);
set(plotID, 'Position', [1 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
%plot(z_pred, 'r');
title('True state (blue) and Estimated state (red)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStateEstimates');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 1002;
figure(plotID);
set(plotID, 'Position', [500 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(X_k, 'b');
plot(XX_k1k1, 'r');
plot(Z_k, 'k');
title('True state (blue), Estimated state (red), Measured state (black)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2001;
figure(plotID);
set(plotID, 'Position', [800 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
title('State estimation error');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end

plotID = 2002;
figure(plotID);
set(plotID, 'Position', [800 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
axis([0 50 min(EstErr) max(EstErr)]);
title('State estimation error (Zoomed in)');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2003;
figure(plotID);
set(plotID, 'Position', [1000 100 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
title('State estimation error with STD of Innovation');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end


plotID = 2004;
figure(plotID);
set(plotID, 'Position', [1000 550 600 400], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'PaperPositionMode', 'auto');
hold on;
plot(EstErr, 'b');
plot(STDx_cor, 'r');
plot(-STDx_cor, 'g');
axis([0 50 min(EstErr) max(EstErr)]);
title('State estimation error');
legend('Estimation error', 'Upper error STD', 'Lower error STD', 'Location', 'northeast');
if (printfigs == 1)
    fpath = sprintf('fig_demoKFStatesEstimatesMeasurements');
    savefname = strcat(figpath, fpath);
    print(plotID, '-dpng', '-r300', savefname);
end