function [model] = highdim_opt(yamlfile, obj)

%% Read parameters settings.
lambda = 1;
[params, runConfig] = readConfig(yamlfile);        % Read in parameter settings.
high_dim = params.high_dim;                     % Dimensionality of the problem.

map_func = @(x)generic_map(x, params);

dim = runConfig.embeddingDim;
total_iter = runConfig.numIter;
noise = runConfig.noise;

% Setup directory for saving partial results.
savePath = runConfig.saveFolder;
file_path_mat = sprintf('/embedding_dim_%d_iteration_%d.mat', ...
        dim, total_iter);
file_path_mat = strcat(savePath, file_path_mat);

file_path_txt = sprintf('/embedding_dim_%d_iteration_%d.txt', ...
        dim, total_iter);
    file_path_txt = strcat(savePath, file_path_txt);

%% Setup Rembo with high-dimensional kernel.
if high_dim > dim
    A = randn(high_dim, dim);
    scale = max(1.5*log(dim), 1.5);
else
    A = eye(high_dim, dim);
    scale = 1;
end

% Bounds 
bounds = scale*stardardBounds(dim);

% Save A the random projection matrix.
save(sprintf(file_path_mat, dim, total_iter), 'A');

% Wrap the objective function.
obj_fct = @(x) obj_wrapper(obj, (A*x')', map_func);

% For saving results.
partial_result_func = @(model)save_progressive_results(model, ...
    total_iter, obj_fct);

% Sample the first point randomly.
fprintf('Start running..\n');
init_pt = rand(1, dim)*2-1;
[init_f, record] = obj_fct(init_pt);

cate_indices = params.categorical_indices;

if ~runConfig.useHighDimKernel && isempty(cate_indices)
    % Use low-dimensional kernel. Cannot use low-dimensional kernel when
    % there's categorical parameters present.
    hyp = [ones(dim, 1)*lambda*0.05 ; 1];
    hyp = log(hyp);

    model = init_model(dim, bounds, init_pt, init_f, hyp, noise, 'ard');
else
    % Use high-dimensional kernel.
    hyp = [ones(high_dim, 1)*lambda ; 1];
    hyp = log(hyp);

    ck = @(hyp, x, z, records)hybrid_kernel(hyp, x, z, records, map_func, A, ...
        cate_indices);
    model = init_model(dim, bounds, init_pt, init_f, hyp, noise, ...
        'custom', high_dim, record, ck);
end

%% Start optimization.
tic                                                           % Start Bayes opt.
model = sparse_opt(obj_fct, total_iter-1, model, partial_result_func);                
toc


%% Helper functions.
function save_progressive_results(model, total_iter, obj_fct)
    if mod(model.n, 5) == 0 || model.n == total_iter
        % Save mat file every 5 iterations.
        try
            save(file_path_mat, 'model', '-append');
        catch %#ok<CTCH>
            save(file_path_mat, 'model');
        end
    end

    % Save text result in every iteration.
    save_text_results(file_path_txt, model.X(model.n, :), ...
        model.f(end), max(model.f), model.n, obj_fct);
end

% Save progressive results in text.
function save_text_results(file_path_txt, x, fval, max_fval, cur_iter, obj_fct)
    % Save text results.
    [~, ~, cmd_str] = obj_fct(x);
    fileID = fopen(file_path_txt, 'a');
    fprintf(fileID, 'Cur Iter: %4d  Highest reward: %2.4f  Current: %2.5f\n', ...
        cur_iter, max_fval, fval);
    fprintf(fileID, 'Current Parameter settings: \n');
    fprintf(fileID, cmd_str);
    fprintf(fileID, '\n\n');
    fclose(fileID);
end


function [obj, record, base_cmd_str] = obj_wrapper(obj_fct, x, map_func)
    % Wrap objective function.
    obj = 0;
    [x, record] = map_func(x);

    if nargout > 2 
    	base_cmd_str = sprintf('%f ', x);
    else
    	obj = obj_fct(x);
    end
end


function [x, k] = generic_map(x, param)
    % Map from low to high dimensional space.
    x = min(x, 1.0);
    x = max(x, -1.0);
    x = (x + 1)/2;

    x = bsxfun(@times, x, param.param_range);
    x = bsxfun(@plus, x, param.param_bounds(1, :));
    x(:, param.log_indices) = exp(x(:, param.log_indices));
    x(:, param.discrete_indices) = round(x(:, param.discrete_indices));
    
    if ~isempty(param.discrete_indices)
        x(param.discrete_indices) = min(x(param.discrete_indices), ...
            param.param_bounds(2, param.discrete_indices));
    end


    if nargout > 1
    	k = x;
    	k(param.log_indices) = log(x(param.log_indices));
    	k = bsxfun(@minus, k, param.param_bounds(1, :));
    	k = bsxfun(@rdivide, k, param.param_range);
    	k = k*2-1;
    end
end

function M = hybrid_kernel(hyp, x, z, records, map_func, A, cate_indices)

    % Wrap kernel function.
    if min(size(x) == size(z)) && min(x == z)
        M = exp(hyp(end));
    else
        if size(z, 1) == 1
            if nargin > 5
                z = (A*z')';
            end
            [~, z] = map_func(z);
        end
        
        if (~isempty(cate_indices)) && (size(cate_indices, 2) < size(hyp, 1)-1)
            regular_indices = setxor(1:size(hyp), cate_indices);
            M = covSEard(hyp(regular_indices), ...
                records(:, regular_indices(1:end-1)), ...
                z(:, regular_indices(1:end-1)));
            k2 = categorical_kernel(z(:, cate_indices), ...
                records(:, cate_indices), hyp(cate_indices));
            M = M.*k2;
        elseif isempty(cate_indices)
            M = covSEard(hyp, records, z);
        else
            M = categorical_kernel(z, records, hyp);
        end
    end
end

function [K] = categorical_kernel(z, records, hyp) 
    if size(z, 1) == 1
        k_pre = sum((abs(bsxfun(@minus, records, z)) > 0), 2);
    else
        dim = size(records, 2);
        k_pre = zeros(size(records, 1), size(z, 1));
        
        for d = 1:dim
            k_pre = k_pre + bsxfun(@ne, records(:, d), z(:, d)');
        end
    end
    
    K = exp(-k_pre/exp(hyp(1)));
end

end