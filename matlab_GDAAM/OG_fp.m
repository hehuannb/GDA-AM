function [fp, oldg] = OG_fp(x,oldg,A,b, c, n,lr)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
 x0 = x(1:n);
 y0 = x(n + 1 : end);
 OG_gx = oldg(1:n);
 OG_gy = oldg(n+1:end);
 gx = A * y0 + b ;  
 gy = A'* x0 + c;
 x1 = x0 - lr * gx + lr/2 * OG_gx ;
 y1 = y0 + lr * gy - lr/2 * OG_gy ;  
 fp = [x1;y1];
 oldg = [gx;gy];
end
