function [acq, grad] = acq(model, x, type, si)

if nargin < 4
	si = 1;
end

if nargin < 3
	type = 'ucb';
end
compute_gard = 0;

if nargout >= 2
	compute_gard = 1;
end

[mu, var] = mean_var(model, x);
sigma = sqrt(var);

if compute_gard
	[k_t, cov_grad] = model.cov_model(model.hyp, model.X(1:model.n,:), x);
	intermediate = (model.L'\(model.L\cov_grad'))';

	mu_grad = intermediate*model.f;
	sigma_grad = -intermediate*k_t/sigma;
end

% type
if strcmp('ucb', type)
    coeff = si*sqrt(2*log(model.n^(model.d/2+2) *pi^2 /(3*0.1)));
	acq = -(mu+coeff*sigma);
	
	if compute_gard
		grad = -(mu_grad + coeff*sigma_grad);
	end
elseif strcmp('ei', type)
	% diff = (mu - model.max_val-0.01);
	% Z = diff./sigma;
 	% [npdf, ncdf] = stand_normal_stats(Z);
	% acq = -(ncdf.*diff + npdf.*sigma);

	% if compute_gard
	% 	Z_grad = (mu_grad*sigma - sigma_grad*diff)/var;
	
	% 	grad = ncdf*mu_grad + diff*npdf*Z_grad;
	% 	grad = grad + npdf*sigma_grad + sigma*npdf*(-Z)*Z_grad;
	% 	grad = -grad/model.noise;
	% end

	acq = -log_exp_imp(-model.max_val, -(mu-0.0001), sigma);
end


function [npdf, ncdf] = stand_normal_stats(Z)
	npdf = (2*pi)^(-0.5)*exp(-(Z.^2)/2);
	ncdf = 0.5*(1+erf(Z/sqrt(2)));
end

end