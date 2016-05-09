function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

thetaT = theta;
thetaT(1) = 0;

correction = sum(thetaT .^ 2) * (lambda/(2*m));
J = sum(term1 - term2) / m  + correction;

grad = (X' * (sigmoid(X * theta) - y)) + thetaT * (lambda/m);

% =============================================================

end