function log_expected_improvement = log_exp_imp(f_min, mu, sigma)
%sigma(find(sigma<1e-10)) = NaN; % To avoid division by zero.
% Tom Minka's normpdf takes row vectors -- column vectors are 
% interpreted as multivariate.
expected_improvement = (f_min-mu) .* normcdf((f_min-mu)./sigma) + sigma .* normpdf(((f_min-mu)./sigma)')';

%Expected improvement often yields zero. Use more robust log expected improvement
%instead.

x = (f_min-mu)./sigma;
log_expected_improvement = zeros(length(mu),1);
for i=1:length(mu)
    if abs(f_min-mu(i)) == 0
        % Degenerate case 1: first term vanishes.
        if sigma(i) > 0
            log_expected_improvement(i) = log(sigma(i)) + normpdfln(x(i));
        else
            log_expected_improvement(i) = -inf;
        end
    elseif sigma(i) == 0 
        % Degenerate case 2: second term vanishes and first term has a special form.
        if mu(i) < f_min
            log_expected_improvement(i) = log(f_min-mu(i));
        else
            log_expected_improvement(i) = -inf;
        end
    else 
        % Normal case.
        b = log(sigma(i)) + normpdfln(x(i));
        % log(y+z) is tricky, we distinguish two cases:
        if f_min>mu(i)
            % When y>0, z>0, we define a=ln(y), b=ln(z).
            % Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
            % and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
            a = log(f_min-mu(i)) + normcdfln(x(i));
            log_expected_improvement(i) = max(a,b) + log(1 + exp(-abs(b-a)));
        else
            % When y<0, z>0, we define a=ln(-y), b=ln(z), 
            % and it has to be true that b >= a in order to satisfy y+z>=0.
            % Then y+z = exp[ b + ln(exp(b-a) -1) ],
            % and thus log(y+z) = a + ln(exp(b-a) -1)
            a = log(mu(i)-f_min) + normcdfln(x(i));
            if a >= b 
            % a>b can only happen due to numerical inaccuracies or 
            % approximation errors
                log_expected_improvement(i) = -inf;
            else
                log_expected_improvement(i) = b + log(1-exp(a-b));
            end
        end
    end
end
log_expected_improvement(find(log_expected_improvement==-inf)) = -1e100;
