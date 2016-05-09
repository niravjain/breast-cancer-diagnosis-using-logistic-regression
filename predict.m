function [p F1] = predict(theta, X, y)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta. Also return the F1 score computed from
%the predictions
%   [p F1] = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1) and
%   returns the F1 score for better analysis of skewed classes

% =========================================================================

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

% We make predictions using our learned logistic regression parameters. 
% We set p to a vector of 0's and 1's

p = round(sigmoid(X * theta));

predictions = (p >= 0.5);
tp = sum(predictions == 1 & y == 1);
fp = sum(predictions == 1 & y == 0);
fn = sum(predictions == 0 & y == 1);

prec = tp / (tp + fp);
rec = tp / (tp + fn);

F1 = (2 * prec * rec) / (prec + rec);

% =========================================================================

end
