% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: rk4.m
% NOTE: Runge-Kutta function -> confirmed in: 
% http://mathworld.wolfram.com/Runge-KuttaMethod.html
function [t,w] = rk4(fn, F, B, xin, uin, t)
    a = t(1); 
    b = t(2);
    w = xin;
    N = 2;
    h = (b-a) / N; % intermediate point
    t = a; % initial time

    for j=1:N
        K1 = h * fn(t, w, uin,F,B);
        K2 = h * fn(t+h/2, w+K1/2, uin,F,B);
        K3 = h * fn(t+h/2, w+K2/2, uin,F,B);
        K4 = h * fn(t+h, w+K3, uin,F,B);

        w = w + (K1 + 2*K2 + 2*K3 + K4) / 6;
        t = a + j*h;
    end
