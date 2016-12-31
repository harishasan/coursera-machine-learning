function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X = 47 X 3
% Y = 47 X 1
% theta = 3 X 1
% X_times_theta = 47 X 1
X_times_theta = X * theta;
% X_times_theta_minus_y = 47 X 1
X_times_theta_minus_y = X_times_theta - y;
% transpose_of_X_times_theta_minus_y = 1 X 47
transpose_of_X_times_theta_minus_y = (X_times_theta_minus_y)';
% 1 x 47 * 47 X 1 = [1]
J = 1/(2 * m) * (transpose_of_X_times_theta_minus_y * X_times_theta_minus_y);
% =========================================================================

end