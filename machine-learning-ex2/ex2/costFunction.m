function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% The original foruma says sigmoid(theta ' * x). 
% I am not entirely sure why this is working
% I know probably the rule being applied is theta ' * X = X' * theta
% but we do not have X' rather we are using simple X. 
% X = 20 X 3
% theta = 3 X 1
% cost = 20 X 1
cost = sigmoid(X * theta);
% again we need to use y' to match dimenstions
% y = 20 X 1
J = (1/m) * sum((-y' * log(cost)) - ((1 -  y)' * log(1 - cost)));
x_tranpose_times_cost_minus_y = X' * (cost - y);
grad = (1/m) .* x_tranpose_times_cost_minus_y;
% =============================================================
end
