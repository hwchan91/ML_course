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


a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2); % returns m x 25 matrix
a2 = [ones(m,1) a2]; % returns m x 26 matrix

z3 = a2 * Theta2';
a3 = sigmoid(z3); % returns m x 10 matrix
hx = a3;

y_binarized = sparse (1:rows (y), y, 1); % convert y vector into a m x K matrix consisting of 1 and 0s
regJ_matrix = (-y_binarized .* log(hx) - ((1 - y_binarized) .* log(1 - hx))) ./ m;
regJ = sum(regJ_matrix(:));

Theta1_variance = Theta1(:, 2:input_layer_size + 1) .^ 2;
Theta2_variance = Theta2(:, 2:hidden_layer_size + 1) .^ 2;
J = regJ + lambda / (2 * m) * (sum(Theta1_variance(:)) + sum(Theta2_variance(:)));

delta3 = a3 .- y_binarized; % m x K matrix
delta2 = delta3 * Theta2(:, 2:hidden_layer_size + 1) .* sigmoidGradient(z2); % delta3(m x K), Theta2 without 1s(K * hidden_layer_size), sig_grad2 (m x hidden_layer_size), resulting in m x hidden_layer_size matrix
D2 = delta3' * a2; % delta3 (m x K), a2 (m * hidden_layer_size+1), resulting in K x hidden+1 matrix
D1 = delta2' * a1; % delta2 (m * hidden_layer_size), a1 (m x var+1), resulting in hidden_layer_size x variables+1 matrix

Theta1_copy = Theta1;
Theta1_copy(:,1) = 0;
Theta2_copy = Theta2;
Theta2_copy(:,1) = 0;

Theta2_grad = D2 / m + lambda / m * Theta2_copy;
Theta1_grad = D1 / m + lambda / m * Theta1_copy;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
