function plot_FF_inputs(Y,num_fig,plot_title,Z_k,Z_k_train,Z_k_val,Z_k_test,Y_train,Y_val,Y_test) % Y = Cm 
%%   Plotting raw alpha and beta data-points - FF
%---------------------------------------------------------

figure(num_fig);
% set(plotID, 'Position', [0 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'color', [1 1 1], 'PaperPositionMode', 'auto');

% Plot data points
if nargin == 4 
    plot3(Z_k(:,1), Z_k(:,2), Y, '.k'); 
else 
    plot3(Z_k_train(:,1), Z_k_train(:,2), Y_train, '.b'); 
    hold on;
    plot3(Z_k_val(:,1), Z_k_val(:,2), Y_val, '.g'); 
    hold on;
    plot3(Z_k_test(:,1), Z_k_test(:,2), Y_test, '.r');
end 

view(0, 90); 
ylabel('beta [rad]');
xlabel('alpha [rad]');
zlabel('C_m [-]');
title(['F16 (\alpha, \beta) - ' plot_title]);
if nargin > 4 
    legend('Training data','Validation data','Testing data');
end 

end