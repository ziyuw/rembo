function y = branin(x, original)
% 
% Branin function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
% 

	if nargin < 2
		original = 0;
	end

	if ~original
		bounds_branin = [-5,10; 0,15];
		x(1:2) = bounds_branin(1:2, 1)' + ...
			((x(1:2)+1)/2).*(bounds_branin(1:2, 2) - bounds_branin(1:2, 1))';
	end

	y = (x(2)-(5.1/(4*pi^2))*x(1)^2+5*x(1)/pi-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;
end