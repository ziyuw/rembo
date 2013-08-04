function model = update_model(model, f_t, final_xatmin)

% decide whether the model has been exploting too much.
[m , v] = mean_var(model, final_xatmin);
if sqrt(v) <= 2*sqrt(model.noise)
    model.exploit_count = model.exploit_count + 1;
else
    model.exploit_count = 0;
end

% Book keeping.
model.n = model.n + 1;
model.X(model.n,:) = final_xatmin;
model.f = [model.f', f_t]';

if f_t > model.max_val
    model.max_x = final_xatmin;
    model.max_val = f_t;

    
end

if model.display % && model.n - 1 > model.last_display + 5
        fprintf('Iteration: %3i:  Max Value: %e\n', ...
            model.n-1, model.max_val);
        model.last_display = model.n-1;
    end

if model.n-1 >= 20 && mod(model.n-1, 20) == 0
    % Routine optimization of hyper-parameters every 20 iterations.
    learn_hyper = 1;
else
    learn_hyper = 0;
end

if model.exploit_count >= model.max_expoit_count
    % If BO has been exploiting too much then lower upper bound of
    % hyper-parameters.
    model.hyper_bound = [model.hyper_bound(1), ...
        model.cur_hyp+log(0.9)];
    model.exploit_count = 0;
    learn_hyper = 1;
end

if learn_hyper
    % Learn hyper-parameters.
    new_hyp = learn_hyper_param(model);
    model.cur_hyp = new_hyp(1);
    model.hyp(1:end-1, 1) = new_hyp;
end

model = update_kernel(model, final_xatmin, learn_hyper);        % Update Kernel.

%% Update Kernel after seeing new data.
function model = update_kernel(model, x, learn_hyper)
    m = model.m;
    k_x = model.cov_model(model.hyp, model.X(1:m,:), x, ...
        model.records(1:min(model.m, size(model.records, 1)),:));
    k_tt = model.cov_model(model.hyp, x, x, ...
        model.records(1:min(model.m, size(model.records, 1)),:));

    m = m + 1;
    model.m = m;

    if ~learn_hyper
        % If not learning Hyper-parameter then update the kernel matrix.
        z_t = model.L\k_x;
        d_t = sqrt(k_tt + model.noise - z_t'*z_t);
        model.L = [[model.L, zeros(m-1, 1)];[z_t', d_t]];
    else
        % If learn hyper-parameter than recompute kernel matrix.
        if strcmp(model.kernel_type,'custom')
            model.L = model.cov_model(model.hyp, 0, ...
                model.records(1:model.m,:), model.records(1:model.m,:)) + ...
                eye(model.n)*model.noise;
        else
            model.L = model.cov_model(model.hyp, ...
                model.X(1:m,:), model.X(1:m,:)) + eye(model.m)*model.noise;
        end

        % Do Cholesky decomposition.
        model.L = chol(model.L, 'lower');
    end
end

%% log_marginal_likelihood calculations.
function new_hyp = learn_hyper_param(model)
    
    dopt.maxevals = 400;
    dopt.maxits = 300;
    dopt.showits = 0;
    
    s = size(model.hyp, 1)-1;
    problem.f = @(x) log_marginal_likelihood([ones(s, 1)*x; model.hyp(end)], ...
        model);

    % Learn Hyper-parameters by using direct.
    [~, x_max, ~] = direct(problem, model.hyper_bound, dopt);
    new_hyp = ones(s, 1)*x_max;
end


function [lml] = log_marginal_likelihood(hyper_param, model)
    % Log marginal likelihood.
    if strcmp(model.kernel_type,'custom')
        % Compute kernel matrix.
        K = model.cov_model(hyper_param, 0, model.records(1:model.n,:), ...
            model.records(1:model.n,:)) + eye(model.n)*model.noise;
    else
        % Compute kernel matrix.
        K = model.cov_model(hyper_param, model.X(1:model.n,:), ...
            model.X(1:model.n,:)) + eye(model.n)*model.noise;
    end

    Kchol = chol(K);
    ALPHA = (Kchol\(Kchol'\model.f));
    lml = model.f'*ALPHA + 2*sum(log(diag(Kchol)));
end

end
