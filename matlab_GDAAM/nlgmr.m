function [xnew, Dat] = nlgmr(x, Dat, g) 
  %%-------------------- unpack
  %% DP stores the p's 
  %% DF stores the w's
  %% 
  nv = Dat.nv;
  DP = Dat.DX;
 %%--------------------residuals
  DW = Dat.DF;
  F = @(x)(x-g(x));
%%beta = data.beta;
%%-------------------- new x,f
%% f = gx-x;
%%-------------------- very first time. 
 [n, m] = size(DP);
 if (m == 0)
   f  = F(x);
   DP = f;
   m  = 1;
 else
   f  = DW(:,m);
 end
%%-------------------- line search
 p = DP(:,m); 
 [xnew, alp, fnew, wj] =  lsearch(F, p, x, f);
 wj = -(fnew -f);
 t = sqrt(p'*wj);
 wj = wj/alp; 
 p  = fnew;
 DP(:,m)=p;
 %%%
 DW(:,m) =wj;
%% [alp, norm(xnew-x,1)]
 pnew = fnew;
 %%-------------------- get new p (p_[j+1]) 
 if (0)
%%   tau = 1.e-03;
%%   pnew = F(x+tau*fnew);
 bet = DW(:,1:m)'*pnew;
 G = DW(:,1:m)'*DW(:,1:m);
 bet = G \ bet;
 pnew = pnew - DW(:,1:m)*bet;
else
  for k=1:m
    wk   = DW(:,k);
%%--------------------FIX-ME -- need to save norm^2 of pk's!
    bet  = fnew'*wk/ (wk'*wk);
        pnew = pnew - bet * DP(:,k);
%     bet  = pnew'*wk;
%     pnew = pnew - bet * DP(:,k);
 end
 end
% pnew = pnew/norm(pnew);

%% pnew'*DW ./ sqrt(sum( DW .* DW, 1)) 
 %%-------------------- store
 m1 = m+1;
%%-------------------- save f for next time 
 DW(:,m1) = f;
%%-------------------- update DP and DW -- most recent x, f were stored
%%                     in m-th columns of DP and DW
 DP(:,m1)  = pnew;
%%-------------------- use truncated svd or some other regul.
%%-------------------- move down all vectors when size exceeds nv
 if (m >= nv) 
   DW = DW(:, 2:end);
   DP = DP(:, 2:end);
 end
 Dat.DX = DP;
 Dat.DF = DW;
 %%-------------------- 
 
