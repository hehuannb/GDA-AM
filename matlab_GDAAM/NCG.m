function [w,iter,res_hist, rest_hist] = NCG(g, solution, w, step, itmax, tol, print)

res_hist = []; 
rest_hist = [];
tic
grad = g(w);
grad_old = g(w);         
d_old = -g(w);
d = length(w); 
S = eye(d);
for iter = 0:itmax    
    if (d_old'*grad_old > 0)
        d_old = -d_old;
    end      
    w_old = w;        
    w = w + step * S * d_old;
    grad_old = grad;   
    % calculate gradient
    grad = g(w);    
     beta = ((grad-grad_old)'*S*grad)/(grad_old'*S*grad_old);
    if beta < 0
        beta = max(0,beta);
    end   
    d = -S * grad + beta * d_old;    
    
    % store d
    d_old = d;
    if mod(iter, print) ==0 
        res_norm = norm(w - solution);
        fprintf('%d %e \n', iter, res_norm);
        res_hist = [res_hist;[iter,res_norm]];
        rest_hist = [rest_hist;[toc,res_norm]];
    end
    
    if res_norm <= tol || isnan(res_norm)
        fprintf('Terminate with residual norm = %e \n\n', res_norm);
        break;
    end
    
end

if res_norm > tol && iter == itmax
    fprintf('\n Terminate after itmax = %d iterations. \n', itmax);
    fprintf(' Residual norm = %e \n\n', res_norm);
end
