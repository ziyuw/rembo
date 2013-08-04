function model = init_model(d, bounds, init_pt, init_f, hyp, noise, ...
    cov_model, high_dim, init_record, custom_kernel)

model.use_direct = 1;
model.use_CMAES = 0;
model.use_fminsearch = 1;
model.use_gradient = 0;

model.noise = noise;

if strcmp(cov_model,'se')
    model.cov_model = @(hyp, x, z, records)covSEiso(hyp, x, z);
elseif strcmp(cov_model,'ard')
    model.cov_model = @(hyp, x, z, i)covSEard(hyp, x, z);
elseif strcmp(cov_model,'custom')
    model.cov_model = @(hyp, x, z, records)custom_kernel(hyp, x, z, records);
elseif strcmp(cov_model,'custom_simple')
    model.cov_model = @(hyp, x, z, records)custom_kernel(hyp, x, z);
end

model.kernel_type = cov_model;

model.hyp = hyp;
model.bounds = bounds;
model.d = d;

model.cur_hyp = hyp(1);
model.exploit_count = 0;
model.hyper_bound = log([0.01, 50]);

model.prior_mean = 0;

model.max_expoit_count = 5;

if nargin < 8
    model.records = 0;
    model.high_dim = 0;
else
    model.records = zeros(3000, high_dim);
    model.records(1, :) = init_record;
    model.high_dim = high_dim;
end

model.L = (model.cov_model(model.hyp, init_pt, init_pt, model.records) + ...
    model.noise);
model.L = chol(model.L, 'lower');

model.X = zeros(3000, d);
model.X(1, :) = init_pt;

model.f = init_f;

model.m = 1;
model.n = 1;

model.max_val = init_f;
model.max_x = init_pt;
model.display = 1;


model.copts.MaxFunEvals = 2000; model.copts.TolFun = 1e-8; 
model.copts.LBounds = model.bounds(:, 1);
model.copts.UBounds = model.bounds(:, 2); 
model.copts.SaveVariables = 0; model.copts.LogModulo = 0;
model.copts.DispFinal = 0; model.copts.DispModulo = 0;


model.last_display = 0;