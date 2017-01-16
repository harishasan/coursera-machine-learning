function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% get, against each input row, vector of K size, where each cell in the vector represents proability of each class/label
% in other words, against each input row, calculate the probability of each output/label using trained Theta1 and Theta2. 


% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% calculate h(x) using particular values of Theta1 and Theta2
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
z2 = X * Theta1';
a2 = sigmoid(z2);
% add ones columns again in new X i.e. a2
a2 = [ones(size(a2)(1), 1) a2];
h_of_x = sigmoid(a2 * Theta2');

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% convert y into matrix
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% y is a one dimensional vector containing 1-10 digits.
% lets convert it into a matrix where each output is represented as a vector e.g 9 would look like this [0   0   0   0   0   0   0   0   1  0]
yIdentity = eye(num_labels);
y = yIdentity(y,:);
inner_cost = -y .* log(h_of_x) - ((1 - y ) .* log(1 - h_of_x));

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% compute cost of J(theta)
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% cost function without regularization
%sum(sum()) is summing a 2D matrix
% J = ((1/m) .* sum(sum(inner_cost))); 

% cost function with regularization
J = ((1/m) .* sum(sum(inner_cost))); 

%without bias because we do not apply regularization on bias unit
theta_1_without_bias = Theta1(:, 2:size(Theta1)(2));
theta_2_without_bias = Theta2(:, 2:size(Theta2)(2));

J = J + (lambda/(2 * m) * (sum(sum(theta_1_without_bias.^2)) + sum(sum(theta_2_without_bias.^2))));
% -------------------------------------------------------------

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Step1: forward propagation
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

% we know a1 = X and a3 is output
% a2 = sigmoid(X * Theta1'); already calculating above
% add bias column
% a2 = [ones(size(a2)(1), 1), a2];
% a3 has same number of columns as output/labels. It contains, against each input row, proability of each output class
a3 = h_of_x; %sigmoid(a2 * Theta2'); already doing above

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Step 2,3: calculating errors using backpropagation
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

%output_error(a3 error) = actual_output - our_output
layer3_or_a3_errors = a3 - y;
z2 = [ones(m,1) z2];
layer2_or_a2_errors = layer3_or_a3_errors * Theta2 .* sigmoidGradient(z2);
%removing bias unit error
layer2_or_a2_errors = layer2_or_a2_errors(:,2:end);

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Step 4: calculate error in current thetas
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

theta_2_errors = layer3_or_a3_errors' *a2;
% X = a1
theta_1_errors = layer2_or_a2_errors' *X; 

% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
% Step 5: calculate gradient and also do regularization 
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

theta_1_regularized = (lambda/m) * Theta1(:, 2: end);
theta_2_regularized = (lambda/m) * Theta2(:, 2: end);
theta_1_regularized_with_zeros = [zeros(size(theta_1_regularized)(1), 1), theta_1_regularized];
theta_2_regularized_with_zeros = [zeros(size(theta_2_regularized)(1), 1), theta_2_regularized];
Theta1_grad = (1/m) .* theta_1_errors + theta_1_regularized_with_zeros;
Theta2_grad = (1/m) .* theta_2_errors + theta_2_regularized_with_zeros;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
