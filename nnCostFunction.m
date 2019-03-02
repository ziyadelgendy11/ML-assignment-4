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


a1 = [ones(m,1) , X];
z2 = Theta1 * a1';
a2 = sigmoid(z2);
size(a2,1)
size(a2,2)
z3 = Theta2 * [ones(m,1) , a2']';
a3 = sigmoid(z3);
h = a3;


completeY = zeros(m , num_labels);
for(i=1:1:m)
    for(j=1:1:num_labels)
       if(y(i) == j)
           completeY(i,j) = 1;
       end
    end
end


J = (1/m) * ( (-completeY .* log(h)') - ( (1-completeY) .* log(1 - h)' ) );
J = sum(sum(J));





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


theta1WithoutBias = Theta1(2:end);
theta2WithoutBias = Theta2(2:end);
newRegularizationPart = sum(sum(theta1WithoutBias.^2)) + sum(sum(theta2WithoutBias.^2));
newRegularizationPart = (lambda / (2*m)) * newRegularizationPart;

J = J + newRegularizationPart;

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for(i=1:1:m)
  current_a1 = a1(i,:);  
  current_z2 = z2(:,i);
  current_a2 = [1;a2(:,i)];
  current_z3 = z3(:,i);
  current_a3 = a3(:,i);
  current_completeY = completeY(i,:); 
  current_z2 = [1; Theta1 * current_a1'];
  errorOutputLayer = current_a3' - current_completeY; 
  errorHiddenLayer = Theta2' * errorOutputLayer' .* sigmoidGradient(current_z2);
  
  
  delta1 = delta1 + (errorHiddenLayer(2:end) * current_a1);
  delta2 = delta2 + errorOutputLayer' * current_a2';
  
end

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

part1Delta1 = delta1(1:end,1);
part2Delta1 = delta1(1:end,2:end);
part1Delta1 = part1Delta1 * (1/m);
part2Delta1 = part2Delta1 * (1/m) + (lambda/m) * Theta1(1:end,2:end);


part1Delta2 = delta2(1:end,1);
part2Delta2 = delta2(1:end,2:end);
part1Delta2 = part1Delta2 * (1/m);
part2Delta2 = part2Delta2 * (1/m) + (lambda/m) * Theta2(1:end,2:end);


Theta1_grad = [part1Delta1 , part2Delta1];
Theta2_grad = [part1Delta2 , part2Delta2];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
