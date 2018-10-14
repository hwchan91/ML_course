function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% options = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% n = numel(options);
% C_sigma_map = zeros(n^2, 3);
%
% for i = 1:n
%   for j = 1:n
%     C_i = options(i);
%     sigma_j = options(j);
%     model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
%     predictions = svmPredict(model, Xval);
%     C_sigma_map(n * (i - 1) + j, :) = [C_i, sigma_j, mean(double(predictions ~= yval));];
%   end
% end
%
% [val, col] = min(C_sigma_map(:,3));
% C = C_sigma_map(col,1);
% sigma = C_sigma_map(col,2);

C = 1;
sigma = 0.1;

% =========================================================================

end
