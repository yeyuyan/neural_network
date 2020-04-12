clear all;
close all;
clc;

% Demonstration du probleme du reglage des hyper-parametres
X = 1:0.5:10;
f = @(x) x+.3*cos(3*pi*x);
Y = f(X);

figure;
plot(X,Y,'r*');
Lgd = {'Data'};
hold on;
TailleHidden = [1 5 10 20];

for i = 1:length(TailleHidden)
    clear net;
    net = feedforwardnet(TailleHidden(i));
    net.performParam.regularization = 0.1;
    net.Layers{1}.transferFcn = 'tansig';
    net.Layers{1}.initFcn = 'initnw';
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2; 
    net.divideParam.testRatio = 0;
    [net] = train(net,X,Y);
    %view(net);
    
    %Visualisation de la capacite a interpoler & extrapoler
    dx = max(X)-min(X);
    X2 = linspace(min(X)-dx/3,max(X)+dx/3,100);
    yest = net(X2);
    plot(X2,yest);
    Lgd{end+1} = sprintf('%i neurones',TailleHidden(i));
    legend(Lgd);
    title(sprintf('\\lambda : %.1g',net.performParam.regularization));
    
end
grid on;
    
 