function [sol, res, cost, P,AP]= nltgcr(FF,sol,lb,tol,itmax, sol_opt, problem, restart)  
%% function [sol, res, P,AP]= nltgcr(FF,sol,lb,tol,itmax)  
%% see dqgmres for meaning of parameters. 
%% truncated GCR/Orthomin -- Note lb does not quite have same meaning
%% as in dqgmres (we use lb 'p' vectors).
%%----------------------------------------------------------------------- 
%%-------------------- initialize
 epsf = 1.e-4;
 n = length(sol); 
 P  = zeros(n,lb);
 AP = zeros(n,lb);
%%--------------------get initial residual vector and norm 
 r = FF(sol);   %% y - A*sol;
 rho = norm(r);
 tol1 = tol*rho; 
 %%-------------------- Ar + normalize    FF = b-Ax --> 
 ep = epsf*norm(sol)/rho;
 
 Ar = (FF(sol-ep*r) - r)/ep;
 t = norm(Ar);
 t = 1.0 / t;
 P(:,1) = r/t;
 AP(:,1) = Ar/t;    
 it = 0;
 res(1) = problem.cost(sol);
 cost(1) = norm(sol-sol_opt);
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
   sol = sol+alph*P(:,i2);
   Fc  = FF(sol);
   r   = r - alph*AP(:,i2);
   %%## NOTE: ALTERNATIVE    Fc defined as Fc = r --> one less feval
   %% but no good theoretical support for this. 
   rho = norm(r);
%%--------------------
   res(it+1) = problem.cost(sol);
   dist = norm(sol-sol_opt);
   cost(it+1) = dist;
   if (rho < tol1), break, end
   fprintf(1,' it %d  distance to optimal %10.3e \n', it, dist);
   %%%-------------------- A second fun eval - not sure it can be avoided. 

   ep = epsf*norm(sol)/rho;

   Ar = (FF(sol-ep*r) - Fc)/ep;
%%Ar  = A*r;
%%--------------------orthonormnalize  Ap's
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
         r = FF(sol);   %% y - A*sol;
         rho = norm(r);
         tol1 = tol*rho; 
         %%-------------------- Ar + normalize    FF = b-Ax --> 
         ep = epsf*norm(sol)/rho;
         
         Ar = (FF(sol-ep*r) - r)/ep;
         t = norm(Ar);
         t = 1.0 / t;
         P(:,1) = r*t;
         AP(:,1) = Ar*t;    
     end
   end
   i2=i2+1;
   AP(:,i2) = Ar/t;
   P(:,i2) = p/t;
 end
