clear all
close all;
clc;

% Calcul symbolique
syms x1 x2 real
f = x1*exp(-x1^2-x2^2);
g = gradient(f,[x1 x2]);

%Creation de fonctions
g1 = symfun(g(1),[x1,x2]);
g2 = symfun(g(2),[x1,x2]);
f = symfun(f,[x1,x2]);

%Evaluation numeriques des fcts sur une grille
[X1 X2] = meshgrid(-2:.2:2,-2:.2:2);
F = f(X1,X2);
G1 = g1(X1,X2);
G2 = g2(X1,X2);

%Conversion symbolique => double
F = double(F);
G1 = double(G1);
G2 = double(G2);

%Affichages
figure;
surf(X1,X2,F);
hold on;
xlabel('x_1');
ylabel('x_2');

figure;
contour(X1,X2,F);%tracer contour de meme valeur
hold on;
quiver(X1,X2,G1,G2);%tracer vecteur
xlabel('x_1');
ylabel('x_2');
