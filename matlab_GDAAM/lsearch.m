     function [xn, alp, rn, w] = lsearch(ffun, p, x, r)
%%   function [xn, alp, rn, w] = lsearch(ffun, p, x, r)
       tau = 1.e-05;
       nrm = norm(x,1) / (1 + norm(p,1));
       tau = tau*nrm;
       %% ADD: CHECK that tau>0
       %%-------------------- Get w=Ap.
       w = ffun(x+tau*p)-r;
       %% w = ffun(x+tau*p)-ffun(x-tau*p);
       w =  w /tau;
       %%--------------------sign! 
       %%alp = -r'*w/ (w'*w);
       %%1 alp = -r'*p/ (w'*p);
       %        alp = -r'*p/ (p'*w);
       alp = -r'*p/ (p'*w);
%%-------------------- linesearch -- turned off.
	if (0)
	 alp = 2*alp;
	 npts = 11;
	 tt = linspace(0,1,npts);
	 tt = alp*tt(2:end);
	 d = norm(r);
	 k = 0;
	 for j=1:npts-1
	   tau = tt(j);
	   t1 = norm(ffun(x+tau*p));
	   if (t1 < d)
	     d = t1;
	     k = j;
	   end
	 end
	 alp = tt(k);
       end
%%--------------------  
%        [n,m] = size(p);
%        sign = [-1*ones(1,n/2),ones(1,n/2)];
       xn = x + alp* p;
       rn = ffun(xn);
    
