clear all;
close all;
clc;

syms x1 x2;

%Fct ¨¤ minimiser
f = exp(-x1^2-x2^2)*(x1^2+x2^2);
f = symfun(f,[x1 x2]);

dfdx = gradient(f,[x1 x2]);
d2fdx2 = hessian(f,[x1 x2]);
d2fdx2 = symfun(d2fdx2,[x1 x2]);

%Valeurs propres du Hessien
lambda = simplify(eig(d2fdx2));
lambda = symfun(lambda,[x1 x2]);

%Valeurs num¨¦riques pour affichage
[X1,X2] = meshgrid(-3:.1:3, -3:.1:3);
F = double(f(X1,X2));
figure;
surf(X1,X2,F,'EdgeColor','none');
hold on;
xlabel('x_1');
ylabel('x_2');
hold on;

%Tracer le cercle
theta = 0:0.1:(2*pi + 0.2);
xx = sin(theta);
yy = cos(theta);
ss = double(f(xx,yy));
plot3(xx,yy,ss,'r','linewidth',2);

%Affichage du gradient
figure;
contour(X1,X2,F);
hold on;
[X1,X2] = meshgrid(-2.5:.1:2.5,-2.5:.1:2.5);

g = gradient(f);
gg = g(X1,X2);
G1 = double(gg{1});
G2 = double(gg{2});
quiver(X1,X2,G1,G2);
xlabel('x_1');
ylabel('x_2');