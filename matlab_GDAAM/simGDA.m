function [fp] = simGDA(x,n,lr,A,b,c)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
 x0 = x(1:n);
 y0 = x(n + 1 : end);
 gx = A * y0 + b ; 
 x1 = x0 - lr * gx ;
 gy = A'* x0 + c;
 y1 = y0 + lr * gy ;  
 fp = [-gx;gy];
end

