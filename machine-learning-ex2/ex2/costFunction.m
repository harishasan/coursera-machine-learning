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


% The original foruma says sigmoid = theta ' * x. 
% I am not entirely sure why we should not pass theta' * x here. I understand that matrix multilpication won't work 
% due to dimenstions but can we just pass whatever fits, does it have no implication?
% I guess I am missing something here. 
% X = 20 X 3
% theta = 3 X 1
% cost = 20 X 1
cost = sigmoid(X * theta);
% again we need to use y' to match dimenstions
% y = 20 X 1
J = (1/m) * sum((-y' * log(cost)) - ((1 -  y)' * log(1 - cost)));
% what I do not understand here: if we want to multiply two things and they don't have same dimenstions
% is it ok to take transpose of one to match dimenstions and get the desired results
x_tranpose_times_cost_minus_y = X' * (cost - y);
grad = (1/m) .* x_tranpose_times_cost_minus_y;
% =============================================================
end
