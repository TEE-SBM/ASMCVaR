clear;
close all;
clc;
%% Setting of parameters
ny = 50;
n = 10;
m = 3;
gamma = 1E-7;
%initbeta = ones(n,1);
initbeta = zeros(n,1);
ITER = 5000000;

%% Generate the data for regression
[X,y] = LRdatagen(ny,n,m);

%% Call regression functions
[opt_loss_L0,opt_supp_L0,opt_beta_L0] = L0regress(X,y,m);
[opt_loss_EnvL0,opt_supp_EnvL0,opt_beta_EnvL0,opt_eta_EnvL0] = EnvL0regress(X,y,m,gamma);
[all_loss_PALM,beta_PALM,eta_PALM] = PALMforEnvL0regress(X,y,m,gamma,initbeta,ITER);
final_loss_PALM = all_loss_PALM(ITER);

figure;
plot(1:ITER,all_loss_PALM);
