function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% =============================================================

g = zeros(size(z));

% We compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

denominator =  1 + exp( -z);
g = 1 ./ denominator;

% =============================================================

end
