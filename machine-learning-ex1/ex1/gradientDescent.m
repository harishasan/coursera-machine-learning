function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % vector based approach
    % X = 97 X 2
    % theta = 2 X 1
    % hypothesis =  97 X 1    
    hypothesis = X * theta;
    % difference_hypotheis_and_actual = 97 X 1
    difference_hypotheis_and_actual = hypothesis - y;
    % transpose_of_x = 2 X 97
    transpose_of_x = X';
    % transpose_of_x_times_difference_hypotheis_and_actual = 2 X 1
    transpose_of_x_times_difference_hypotheis_and_actual  = (transpose_of_x * difference_hypotheis_and_actual);
    %gradient = 2 X 1
    gradient = (1/m) * transpose_of_x_times_difference_hypotheis_and_actual;        
    % theta = (2 X 1) - (number * (2 X 1));
    theta = theta - (alpha .* gradient);
    % here we have a new theta on for this iteration
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
