clear all
close all;
clc;

rng(3,'twister');
% Generate a dataset
n=2;
m = 100;
[X,C] = GenereExemple(n,m);

net = perceptron;
net = train(net,X,C);
view(net);
y = net(X);
fprintf('Nombre d''erreurs : %i\n',sum(y-C))

function [X,C] = GenereExemple(n,m)
X = rand(n,m);
A = (rand(1,n)-0.5)*10;
B = -mean(A*X);
C = A*X+B>0;
end
