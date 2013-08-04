function [params, runConfig] = readConfig(yaml_file)

configs = ReadYaml(yaml_file);
names = fieldnames(configs);
num = length(names);


for i=1:num
    if strcmp(names{i}, 'runConfigs')
        runConfig = getfield(configs, names{i});
        names = [names(1:i-1), names(i+1:end)];
        break;
    end
end

num = length(names);
param_bounds = zeros(num, 2);
discrete_indices = zeros(0, 0);
log_indices = zeros(0, 0);
categorical_indices = zeros(0, 0);
for i=1:num
    field = getfield(configs, names{i});
    param_bounds(i, 1) = field.lowerbound;
    param_bounds(i, 2) = field.upperbound;

    if isfield(field, 'integer') && field.integer
        discrete_indices(length(discrete_indices)+1) = i;
    end
    
    if isfield(field, 'categorical') && field.categorical  
        categorical_indices(length(categorical_indices)+1) = i;
    end

    if isfield(field, 'logscale') && field.logscale
        log_indices(length(log_indices)+1) = i;
    end
end

param_bounds(log_indices, :) = log(param_bounds(log_indices, :));


params.high_dim = size(param_bounds, 1);
params.param_bounds = param_bounds';
params.log_indices = log_indices;
params.discrete_indices = discrete_indices;
params.categorical_indices = categorical_indices;
params.param_range = params.param_bounds(2, :) - params.param_bounds(1, :);
params.param_range(discrete_indices) = params.param_range(discrete_indices) + 1;
params.param_bounds(1, discrete_indices) = ...
    params.param_bounds(1, discrete_indices) - 0.5;

end