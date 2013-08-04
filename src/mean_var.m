function [mean, var] = mean_var(model, x)


k_tt = model.cov_model(model.hyp, x, x, ...
    model.records(1:min(model.n, size(model.records, 1)),:));
k_x = model.cov_model(model.hyp, model.X(1:model.n,:), x, ...
    model.records(1:min(model.n, size(model.records, 1)),:));

intermediate = model.L'\(model.L\k_x);
mean = model.prior_mean+intermediate'*(model.f-model.prior_mean);
var = diag(k_tt - k_x'*intermediate);