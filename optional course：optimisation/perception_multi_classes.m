clear all
close all;
clc;

rng(2,'twister');

n = 2; %Nombre de features
m = 1000; %Nombre d'individus
c = 7; %Nombre de classe
[X,C,B,A,We] = GenereExemple(n,m,c);

figure(1);
hold on;
for i = 1:c
    plot(X(1,C==i),X(2,C==i),'*');
end
xx = xlim;
for i=1:length(A)
plot(xx,A(i)*xx+B(i),'linewidth',2);
end

%Apprentissage
W = zeros(c,n+1);
[nErreur,yest] = EvaluePerformance(X,C,W,m);
k=1;
while nErreur>0
    fprintf('Iteration : %i Erreur : %i \n',k,nErreur);
    for i = 1:m
        x = X(:,i)';
        %Estimation de la sortie
        if yest(i)~=C(i)
            %Prediction incorrecte
            iOK = 1:c==C(i);
            inotok = 1:c~=C(i);
            W(iOK,:) = W(iOK,:)+x;
            W(inotok,:)=W(inotok,:)-x;
        end
    end
    [nErreur,yest]=EvaluePerformance(X,C,W,m);
    figure(2);
    hold on;
    plot(k+W(:)*0,W(:),'*');
    drawnow;
    k = k+1;
end
title(sprintf('Iteration: %i Erreur: %i\n',k,nErreur))


function [X,C,B,A,W] = GenereExemple(n,m,nclasses)
%Genere des valeurs de classes entre 1 et nclasses
X = rand(n,m);
X = [X;ones(1,m)]; %rajoute 1 pour le biais
C = NaN(1,m);

for i=1:nclasses-1
    Delta = 1/nclasses;%Largeur de bande
    A(i) = (rand-.5)*Delta;% rand-.5 genere un nombre entre [-0.5 0.5]
    B(i) = Delta*i;
    W(i,:) = [A(i) -1 B(i)];
end

ended = 0;
while ended == 0
    b = abs(W*X);
    inotok = logical(sum(b<Delta/5));
    X(1:2,inotok) = rand(2,sum(inotok));
    ended = all(~inotok)
end

i1 = W(1,:)*X>0;
C(i1) = 1;
for i = 2:nclasses-1
    iok = (W(i-1,:)*X<0)&(W(i,:)*X>0);
    C(iok) = i;
end
i = nclasses-1;
i1 = W(i,:)*X<0;
C(i1) = nclasses;

end

function [Erreurs,y]=EvaluePerformance(X,C,W,m)
Erreurs=0;
[~,y]=max(W*X);% ~ signifie supprimer tous les elements
Erreurs=sum(C~=y);
end

    
