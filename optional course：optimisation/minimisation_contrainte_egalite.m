%clear all;
close all;
clc;

%Minimisation avec contrainte ¨¦galit¨¦
syms x1 x2 l real
f = x1^2 + 2*x1*x2 - 2*x1 + 2*x2^2 -4*x2;
h = (x1 - 15)^2 +(x2 - 15)^2 - 100;
%Lagragien
L = f+l*h;

%Recherche des points stationnaires
dL = gradient(L,[x1 x2 l]);
Stat = solve(dL,[x1 x2 l]);

X1potentiel = double(Stat.x1);
X2potentiel = double(Stat.x2);
Lpotentiel = double(Stat.l);

%Calcule le minimum et l'argmin
f = symfun(f,[x1 x2]);
F = f(X1potentiel,X2potentiel);
[fmin,i] = min(F);

fprintf('Solution : \n');
fprintf('x1: %.2e x2 : %.2e l : %.2e\n', X1potentiel(i), X2potentiel(i), Lpotentiel(i));
fprintf('param¨¨tre de lagrange: %.2e\n', Lpotentiel(i));
fprintf('crit¨¨re : %.2e\n',F(i));

h = symfun(h,[x1 x2]);
[X1 X2] = meshgrid(-50:1:50,-50:1:50);
F = double(f(X1,X2));
H = double(h(X1,X2));

figure;
contour(X1,X2,F,50);
hold on;
fc = fcontour(h,[xlim,ylim],'LevelList',0);
fc.LineWidth = 1;
fc.LineColor = [0 1 0];
plot(X1potentiel(i),X2potentiel(i),'r*');
xlabel('x_1');
ylabel('x_2');
axis square;
