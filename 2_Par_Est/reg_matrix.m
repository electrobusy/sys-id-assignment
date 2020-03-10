% TuDelft - Faculty of Aerospace Engineering
% Systems Identification of Aerospace Vehicles
% Rohan Camlesh Chotalal -> Student Number: 4746317
% File name: reg_matrix.m

% NOTE: Takes into account the Newton Binomial, because we are talking
% about Multivariate Polynomial model definition
function A = reg_matrix(x,y,d)

n = length(x);
n_columns = sum(0:d + 1); % Using a property of the Pascal triangle. 
                          % Because the Newton Binomial has d+1 terms (think in terms 
                          % of the Pascal Triangle, taking into account the number of 
                          % elements in a row of the triangle). So the idea
                          % is to sum all the elements of the Pascal
                          % triangle.
A = ones(n,n_columns); % Regression matrix

j = 1;

for k = 0:d
    for i = 0:k
        C = perm(k)/(perm(i)*perm(k-i)); % Combinations: (k i)
        A(:,j) = C.*x.^(k-i).*y.^i;
        j = j + 1;
    end
end
end