function x_max = maximize_acq(model, dopt, type, si) 
options = [];
if nargin < 3
	type = 'ucb';
end

if nargin < 4
	si = 1;
end

problem.f = @(x) acq(model, x', type, si);

cur_min = inf;
x_max = unifrnd(model.copts.LBounds, model.copts.UBounds);

if model.use_direct
	[fmin_direct, x_max_direct, ~] = direct(problem, model.bounds, dopt);
	if fmin_direct < cur_min
		x_max = x_max_direct;
		cur_min = fmin_direct;
	end
end

if model.use_CMAES
	starting = unifrnd(model.copts.LBounds, model.copts.UBounds);
	[x_max_CMAES, fmin_CMAES, counteval, stopflag, out, bestever] =...
		cmaes(problem.f, starting, ...
		(model.copts.UBounds-model.copts.LBounds)*0.5, model.copts);

	if fmin_CMAES < cur_min
		x_max = x_max_CMAES;
		cur_min = fmin_CMAES;
	end
end

if model.use_fminsearch
	% Use fminsearch to optimize the acquision even further.
	options.MaxFunEvals = 800;
	options.OutputFcn = [];
	[x_max_search, fmin_search, exitflag, output] = ...
		fminsearchbnd(problem.f, x_max, model.bounds(:, 1), ...
		model.bounds(:, 2), options);
	
	if fmin_search < cur_min
		x_max = x_max_search;
		cur_min = fmin_search;
	end
end

if model.use_gradient
	funProj = @(x) max(min(x, model.bounds(:, 2)),  model.bounds(:, 1));
	options.verbose = 0;
	[x_max_grad, fmin_grad] = minConf_SPG(problem.f, x_max, funProj, options);
	if fmin_grad < cur_min
		x_max = x_max_grad;
		cur_min = fmin_grad;
	end
end