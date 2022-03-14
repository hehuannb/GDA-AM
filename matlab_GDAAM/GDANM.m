function [x,iter,res_hist, rest_hist] = GDANM(g, solution, x, itmax, atol, print)

res_hist = []; 
rest_hist = [];
tic
nm = zeros(size(x));
for iter = 0:itmax
    
    if mod(iter, print) ==0 
        res_norm = norm(x - solution);
        fprintf('%d %e \n', iter, res_norm);
        res_hist = [res_hist;[iter,res_norm]];
        rest_hist = [rest_hist;[toc,res_norm]];
    end
    gval = g(x, nm);
    nm = gval - x;
    if res_norm <= atol || isnan(res_norm) || res_norm > 1e4
        fprintf('Terminate with residual norm = %e \n\n', res_norm);
        break;
    end
    x= gval;
end

if res_norm > atol && iter == itmax
    fprintf('\n Terminate after itmax = %d iterations. \n', itmax);
    fprintf(' Residual norm = %e \n\n', res_norm);
end
