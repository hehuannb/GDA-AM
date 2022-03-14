function [fp] = simEGPM(x,nm, n,lr,A,b,c, mom)


 x0 = x(1:n);
 y0 = x(n + 1 : end);
 vx = nm(1:n);
 vy = nm(n + 1 : end);
 gx = A * y0 + b ; 
 xe = x0 - lr * gx ;
 gy = A'* x0 + c;
 ye = y0 + lr * gy ;
 gx = A * ye + b ; 
 x1 = x0 - 0.5*lr * gx + mom * vx ;
 gy = A'* xe + c;
 y1 = y0 + 0.5*lr * gy + mom * vy ; 
 fp = [x1;y1];
end