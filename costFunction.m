function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% ==============================================================================

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%  We compute the cost of a particular choice of theta by setting J to the cost.
%  We also compute the partial derivatives and set grad to the partial
%  derivatives of the cost w.r.t. each parameter in theta
%
%  Note: grad should have the same dimensions as theta

term1 = -1 .* (y .* log(sigmoid(X * theta)));
term2 = 1 .* ((1-y) .* log((1-sigmoid(X * theta))));

J = sum(term1 - term2) / m;

grad = (X' * (sigmoid(X * theta) - y)) * (1/m);

% ==============================================================================

end