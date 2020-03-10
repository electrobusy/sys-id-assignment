% This function determines the permutation of a given integer
function z = perm(x)

if x == 0
    z = 1;
else 
    z = prod(randperm(x));
end
end