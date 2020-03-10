% TuDelft - Faculty of Aerospace Engineering
% Stochastic Aerospace Systems Practical
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: wei_least_squares.m

% Computes the least-squares
function [ls_par,Y_Val] = wei_least_squares(A,Y,R)

% Weighted Least-Squares:
MULT = A' * R^-1 * A;
COV = MULT^-1;
ls_par = COV * A' * R^-1 * Y;

% Validation:
Y_Val = A * ls_par;

end