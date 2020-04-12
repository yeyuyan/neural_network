clear all
close all;
clc;

syms x real

%Fonction ¨¤ ¨¦tudier
f = x^4 -(163*x^3)/10 + (2463*x^2)/25 -(6544*x)/25 +6448/25;

%Calcul de la d¨¦riv¨¦e
dfdx = diff(f,x)
% Recherche des points stationnaires
x_stat = solve(dfdx,x);

%Tracer la fct
X = (2.5:.02:5.7);
f = symfun(f,x);
F = f(X);
F = double(F);

figure;
plot(X,F,'r');
hold on;
plot(x_stat, f(x_stat),'b*');
grid on;
legend('f(x)','pts stationnaires');
