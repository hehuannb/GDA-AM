function [x,it,res_hist, rest_hist] = TGCR(FF, solution, x, lb, itmax, tol, print)

res_hist = []; 
rest_hist = [];
tic


epsf = 1.e-8;
n = length(x); 
P  = zeros(n,lb);
AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
r = FF(x);   %% y - A*sol;
rho = norm(r);
tol1 = tol*rho; 
%%-------------------- Ar + normalize    FF = b-Ax --> 
ep = epsf*norm(x)/rho;

Ar = (FF(x-ep*r) - r)/ep;
t = norm(Ar);
t = 1.0 / t;
P(:,1) = r*t;
AP(:,1) = Ar*t;    
it = 0;
restart =0;
% res_hist(1) = norm(x-solution);
% rest_hist(1) = 0;
%%
fprintf(1,' it %d  rho %10.3e \n', it,rho);
if (rho <= 0.0)
    return
end
%%--------------------get abs residual norm tol from 1st residual norm
%% --main loop: i = loop index. it # steps
%% i2 points to current column in P, AP. Cycling storage used 
 i2=1;
 Fc = r;
 for it =1:itmax 
   alph = dot(r,AP(:,i2));
   x = x+alph*P(:,i2);
   Fc  = FF(x);
   r   = r - alph*AP(:,i2);
   %%## NOTE: ALTERNATIVE    Fc defined as Fc = r --> one less feval
   %% but no good theoretical support for this. 
   rho = norm(r);
%%--------------------

%    if (rho < tol1), break, end
%    fprintf(1,' it %d  distance to optimal %10.3e \n', it, dist);
   %%%-------------------- A second fun eval - not sure it can be avoided. 

   ep = epsf*norm(x)*rho;

   Ar = (FF(x-ep*r) - Fc)/ep;

   p  = r;
   if (i <= lb), k = 0;, else, k=i2;, end
   while(1) 
%% ---------- define next column - circular storage to avoid copying
     if (k  ==  lb), k=0;, end
     k = k+1;
     tau = dot(Ar,AP(:,k));
     p = p-tau*P(:,k);
     Ar = Ar-tau* AP(:,k);
%%---------- update u (last column of current Hess. matrix)
     if (k == i2), break;, end
   end
   t = norm(Ar);
%%-------------------- Now  Ar==Ap. If   Ap == 0 can't advance         
   if (t == 0.0), return; , end
   if (i2  == lb),
     i2=0;
     if restart == 1
         P  = zeros(n,lb);
         AP = zeros(n,lb);
        %%--------------------get initial residual vector and norm 
         r = FF(x);   %% y - A*sol;
         rho = norm(r);
         tol1 = tol*rho; 
         %%-------------------- Ar + normalize    FF = b-Ax --> 
         ep = epsf*norm(x)/rho;
         
         Ar = (FF(x-ep*r) - r)/ep;
         t = norm(Ar);
         t = 1.0 / t;
         P(:,1) = r*t;
         AP(:,1) = Ar*t;    
     end
   end
   i2=i2+1;
   AP(:,i2) = Ar/t;
   P(:,i2) = p/t;


    if mod(it, print) ==0 
        res_norm = norm(x - solution);
        fprintf('%d %e \n', it, res_norm);
        res_hist = [res_hist;[it,res_norm]];
        rest_hist = [rest_hist;[toc,res_norm]];
    
        if res_norm <= tol || isnan(res_norm)
            fprintf('Terminate with residual norm = %e \n\n', res_norm);
            break;
        end
    end
 end




if res_norm > tol && it == itmax
    fprintf('\n Terminate after itmax = %d iterations. \n', itmax);
    fprintf(' Residual norm = %e \n\n', res_norm);
end
