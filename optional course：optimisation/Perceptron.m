clear all
close all;
clc;

rng(3,'twister');
% Generate a dataset
n=2;
m = 100;
[X,C] = GenereExemple(n,m);

% Apprentissage
iP0 = C==0;
iP1 = C==1;
P0 = X(:,iP0);
P1 = X(:,iP1);

%Initialisation des poids
W = zeros(n,1);
b=0;

%Apprentissage
nErreur = EvaluePerformance(X,C,W,b,m);
k = 1;
while nErreur>0
    fprintf('Iteration : %i Erreur £º %i \n',k,nErreur);
    for i = 1:m
        if (W'*X(:,i)+b>0) ~= C(i)
            if C(i) == 1
                W = W+X(:,i);
                b = b+1;
            else
                W = W-X(:,i);
                b = b-1;
            end
        end
    end
    nErreur = EvaluePerformance(X,C,W,b,m);
    k = k+1;
    
    a = -(W(1)+b)/W(2);
    figure(10);
    clf;
    plot(P0(1,:),P0(2,:),'r*');
    hold on;
    plot(P1(1,:),P1(2,:),'bo');
    grid on;
    hold on;
    xx = xlim;
    plot(xx,-(W(1)*xx+b)/W(2),'r','linewidth',2);
    drawnow;
end
                
function [X,C] = GenereExemple(n,m)
X = rand(n,m);
A = (rand(1,n)-0.5)*10;
B = -mean(A*X);
C = A*X+B>0;
end

function Erreurs = EvaluePerformance(X,C,W,b,m)
y = (W'*X+b)>0; %Sortie du perception
Erreurs = sum(C~=y);
end