%% norm1_function: function description

function [out] = norm1_function(theta)
	global y % data samples
	global M % from the LPM (linear in the parameter model) y_hat = M * theta

	% y must be a column vector
	size_y = size(y);
	if size_y(1) < size_y(2)
		y = y';
	end

	r = y - M * theta;

	out = norm(r, 1);
