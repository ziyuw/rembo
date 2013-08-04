%% Define the objective function.
addpath(genpath(pwd()));
cd AC_blackbox_eval/blackbox_eval_source/
config_file = 'config_file.txt';
numRun = 1;
scenario_file = 'scenario-lpsolve-CORLAT-inst.txt';
[func, obj] = define_func_handle(config_file, scenario_file, numRun);
cd ../..

%% Optimize.
yaml = 'lpsolve.yaml';
model = highdim_opt(yaml, obj);