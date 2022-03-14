  function y = fx(x, data)
% function y = fx(x, data)
% one step of gradient descent for regularized logistic regression
    xi = data.xi;
    yi = data.yi;
    alpha = data.alpha;
    lambda = data.lambda;
    m = data.m;
    tmp = xi * x .* yi;
    g = lambda * x + ((-yi ./ (exp(tmp) + 1))' * xi)'/m;
    y = x - alpha * g;
%%
