function plot_Cm(Y_true,Y,Z_k,num_fig,plot_title) % Y = Cm 
%%   Plotting results
%---------------------------------------------------------
% creating triangulation (only used for plotting here)
TRIeval = delaunayn(Z_k(:, [1 2]));

%   viewing angles
az = 140;
el = 36;

% create figures
figure(num_fig);
% set(num_fig,'Position', [800 100 900 500], 'defaultaxesfontsize', 10, 'defaulttextfontsize', 10, 'color', [1 1 1], 'PaperPositionMode', 'auto');
trisurf(TRIeval, Z_k(:,1), Z_k(:,2), Y, 'EdgeColor', 'none'); 
grid on;
hold on;
% plot data points
plot3(Z_k(:,1), Z_k(:,2), Y_true, '.k'); % note that Z_k(:,1) = alpha, Z_k(:,2) = beta, Y = Cm
view(az, el);
ylabel('beta [rad]');
xlabel('alpha [rad]');
zlabel('C_m [-]');
title(['F16 CM(\alpha_m, \beta_m) - ' plot_title]);
% set fancy options for plotting 
set(gcf,'Renderer','OpenGL');
hold on;
poslight = light('Position',[0.5 .5 15],'Style','local');
hlight = camlight('headlight');
material([.3 .8 .9 25]);
minz = min(Y);
shading interp;
lighting phong;
drawnow();
end