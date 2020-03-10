% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: ord_least_squares.m

% Computes the least-squares
function [ls_par,Y_Val] = ord_least_squares(A,Y)

% Least-Squares:
MULT = A' * A;
COV = MULT^-1;
ls_par = COV * A' * Y;

% Alternative - Using MATLAB function pinv (since the previous method gives a non-invertible matrix)
% ls_par = pinv(A)*Y;

% Validation:
Y_Val = A * ls_par;

end