function testBranin(varargin)

run_high_dim = 0;
embedding = 1;
embedding_consecutive = 0;
test_percentage = 0;

total_iter = 500;
high_dim = 25;
rotate = 0;

%% Test the percentage of success
if test_percentage
    dim = 2;                                         % Intrinsic Dimensionality.
    used_dim = 2;                                              % Dimension used.
    
    maximizers = trueMaximizer();
    scale = max(1.5*log(used_dim));
    test_bounds = stardardBounds(dim)*scale;                  % Standard Bounds.    
    
    prct = success_prctg(high_dim, 1000, dim, used_dim, test_bounds, ...
        maximizers);                         % Success Percentage by simulation.
    
    fprintf('Success Percentage by using %d dimensions is approximately: \n%f.\n', ...
        used_dim, prct);
end


%% Random embedding with true intrisic dimensions
%  True maximum is ensured to fall in to the bounds
if embedding
    dim = 2;
    model = rembo(total_iter, dim, high_dim, rotate, 1);
    ditance_log_plot(total_iter, model.f);
end


%% Random embedding with true intrisic dimensions
%  True maximum is NOT ensured to fall in to the bounds
%  Runs are repeated to ensure covering of the true maximum
if embedding_consecutive
    dim = 2;
    rotate = 0;
    run_iter = 125; num_trial = 4;
    f_values = zeros(run_iter*num_trial,1);
    
    for i = 1:num_trial
        model = rembo(run_iter, dim, high_dim, rotate);
        f_values((i-1)*run_iter+1:i*run_iter) = model.f;    
    end    

    ditance_log_plot(run_iter*num_trial, f_values);
end


%% High Dim
if run_high_dim
    rotate = 0;
    model = rembo(total_iter, high_dim, high_dim, rotate, 0, 0);
    
    ditance_log_plot(total_iter, model.f);
end


%% Run rembo.
    function model = rembo(total_iter, dim, high_dim, rotate, force_in_bounds, ...
        embed)
        % total_iter: total number of iterations.
        % dim: embedding dimension.
        % high_dim: ambient dimension.
        % roate: whether to randomly rotate the objective function.
        % force_in_bounds: to force an optimizer in bound by repeatly drawing
        %                  random embedding matrices.
        % embed: Whether to use a randome embedding matrix. (If not 
        %        then we effectively use regular BO.)

        
    
        if nargin < 5
            force_in_bounds = 0;       % Whether to force an optimizer in bound.
        end

        if nargin < 6
            embed = 1;                        % Whether to use embedding or not.
        end

        if rotate
            % Rotate the objective function.
            [rm, ~] = qr(randn(high_dim, high_dim), 0);
        else
            % Do not rotate the objective function.
            rm = eye(high_dim);
        end

        if embed
            % Generate random projection matrix A.
            [in_bounds, A] = test_fall_in_bound(dim, rm);       
            while ~in_bounds && force_in_bounds
                % Ensure that at least one maximizer fall in bound by 
                % generating as many random projection matrix A as needed.
                [in_bounds, A] = test_fall_in_bound(dim, rm);
            end
        else
            % By setting A to be identity we do not use embedding here.
            A = eye(high_dim, dim);
        end

        scale = max(1.5*log(dim), 1);
        bounds = stardardBounds(dim)*scale;                 % Initialize bounds.
        obj_fct = @(x)-branin((A*x')');     % Initialize the objective function.

        init_pt = zeros(1, dim);                                % Initial point.
        init_f = obj_fct(init_pt);                     % Evaluate initial point.

        hyp = [ones(dim, 1)*0.1 ; 1];          % Setup initial hyper-parameters.
        hyp = log(hyp);

        % Initialize model.
        model = init_model(dim, bounds, init_pt, init_f, hyp, 1e-10, 'ard');
        % Do optimization.
        model = sparse_opt(obj_fct, total_iter-1, model);
    end

%% Helper functions.
    function ditance_log_plot(total_iter, fvalues)
        maximazer = trueMaximizer();
        figure;
        dis = zeros(total_iter,1);
        for i =1:total_iter
            dis(i) = -branin(maximazer(:, 1)') - max(fvalues(1:i));
        end
        loglog(1:total_iter, dis); 
    end

    function [in_bounds, A] = test_fall_in_bound(used_dim, rm)
        maximizers = trueMaximizer();
        test_bounds = stardardBounds(2);
        scale = max(1.5*log(used_dim));
        test_bounds = test_bounds*scale;

        [prct, A] = success_prctg(high_dim, 1, 2, used_dim, test_bounds,...
            maximizers, rm);
        in_bounds =  prct;
        
        if in_bounds
            fprintf('At least one maximizer in bounds.\n');
        else
            fprintf('NO maximizer in bounds.\n');
        end
    end


    function [maximizers] = trueMaximizer() 
        bounds_branin = [-5,10; 0, 15];
        maximizers = [pi, -pi, 9.42478; 2.275, 12.275, 2.475];
        maximizers = bsxfun(@minus, maximizers, bounds_branin(:, 1));
        maximizers = bsxfun(@rdivide, maximizers, bounds_branin(:, 2) - ...
            bounds_branin(:, 1))*2-1;
    end

    function [prct, A] = success_prctg(high_dim, num_trial, dim,...
        used_dim, bounds, maximizers, rm)

        total = 0;
        cmbnts = combntns(1:used_dim,dim);
        num_maximizers = size(maximizers, 2);

        for i = 1:num_trial
            indices = 1:high_dim;
            A = randn(high_dim, used_dim);

            if nargin > 6
                A = rm*A;
            end
            fail = 1;

            for j = 1:size(cmbnts, 1)
                for k = 1:num_maximizers
                    true_maximizer = inv(A(indices(1:dim), cmbnts(j, :))) * ...
                        maximizers(:, k);
                    if ~(sum(true_maximizer <= bounds(:,2)) < dim || ...
                        sum(true_maximizer >= bounds(:,1)) < dim)
                        fail = 0;
                    end
                end
            end
            total = total + fail;
        end
        prct = 1 - total/num_trial;
    end
end