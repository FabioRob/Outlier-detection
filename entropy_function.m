% cost_function: Gibbs entropy cost and its gradient
%   applied in residual vector r = y - y_hat
%   where y is data samples vector and y_hat is its prediction

function [H, grad_H] = entropy_function(theta)
	global y % data samples
	global M % from the LPM (linear in the parameter model) y_hat = M * theta

	size_M = size(M);
	n = size_M(1);
	m = size_M(2);

	% y must be a column vector
	size_y = size(y);
	if size_y(1) < size_y(2)
		r = r';
	end

	r = y - M * theta; % evaluate residuals

	D = sum(r.^2); % least sqare cost
	
	if D == 0 % check if it's near to zero?
		H = 0;
	else
		r2_D = r.^2 / D;
		H = -1/log(n) * sum(r2_D .* log(r2_D));
	end

	if nargout > 1 % if gradient is requested, evaluate it
		grad_H = zeros(1,m);
		for j=1:m
			for i=1:n
				grad_H(j) = grad_H(j) + (1 + log(r2_D(i))) * (r(i) * M(i,j) - r2_D(i) * M(:,j)' * r);
			end
		end
		grad_H = 2 / (D * log(n)) * grad_H';
	end