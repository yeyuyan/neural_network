clear all
close all;
clc;
% Y=A.X+B <=> W1*X1+W2*X2+W3=0
% avec W2=-1; W1=A; W3=B

rng(20,'twister');
% 2 perceptron a 1 couche, pour 4 classes

% Générate a linear separable dataset
n = 2; % Nombre de features
m = 500; % Nombre d'individus
c = 8; % Nombre de classe

[X,C,B,A,We]=GenereExemple2(n,m,c);

%Creer un reseau muti-couches 1 couche de taille 2
Inputs = X(1:2,:);
Outputs = C;
hiddenSizes = 3;
net = feedforwardnet(hiddenSizes); %前反馈神经网络
%parametrage des couches
net.Layers{1}.transferFcn = 'tansig';% fct sigmoide pour la couche 'hidden'

%Paramétrage de l’algorithme d’apprentissage
net.divideParam.trainRatio = 1; % trainning set
net.divideParam.valRatio = 0; % validation set
net.divideParam.testRatio = 0; % test set
net.adaptFcn = 'learnpn';

% Apprentissage du réseau sur les données fournies
net = configure(net,Inputs,Outputs);
[net,tr,Y,E] = train(net,Inputs,Outputs);
view(net);

yest = round(net(Inputs));
fprintf('Nombre d''erreurs : %i\n',sum(yest~=C));

%Generation d'une grille X-Y et evaluation du reseau de neuronnes
[X,Y] = meshgrid(linspace(0,1,100),linspace(0,1,101));
inp = [X(:)';Y(:)'];
Z = zeros(size(X));
Z(:) = net(inp);

figure;
subplot(1,2,1);
hold on;
for i =1:c
    plot(Inputs(1,C==i),Inputs(2,C==i),'*')
end
xx = xlim;
yy =ylim;
for i = 1:length(A)
    plot(xx,A(i)*xx+B(i),'linewidth',2);
end
xlim([0 1]);ylim([0 1]);
subplot(1,2,2);
surf(X,Y,Z,'EdgeColor','none');
grid on;
hold on;
for i = 1:c
    plot3(Inputs(1,C==i),Inputs(2,C==i),C(C==i),'*');
end




function [X,C,B,A,W]=GenereExemple2(n,m,nclasses)
%Genere des valeurs de classes entre 1 et nclasses
iter = nextpow2(nclasses);% nextpow2 returns the smallest power of two that is greater than or equal to the absolute value of nclasses
if nclasses~=2^iter
    error('nclasses doit etre de la forme 2^m')
end
ended = 0;
while ended == 0
    X = rand(n,m);
    X=[X;ones(1,m)]; % Rajoute 1 pour le biais
    C=ones(1,m);
    
    for i = 1:iter
        % Droite qui passe par (x1,y1) et (x2,y2)
        x1 = mod(i,2)*rand;
        y1 = rand*0.5+0.25;
        x2 = rand*0.5+0.25;
        y2 = mod(i+1,2)*rand;
        A(i) = (y1-y2)/(x1-x2);
        B(i) = y1-A(i)*x1;
        W(i,:) = [A(i) -1 B(i)]/rand;
        C = C+(W(i,:)*X>0)*2^(i-1);
    end
    for i = 1:nclasses
        cumC(i)=sum(C==i);
    end
    ended = 1;
end
end

        
      