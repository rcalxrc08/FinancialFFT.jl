using FinancialToolbox,DualNumbers
include("carrmadan.jl")


A=400.0;
N=12;

method=CarrMadanMethod(A,N);
S0=100.0;
K=100.0;
r=0.02;
T=1.0;
d=0.01;
sigma=dual(0.2,1.0);
lam=15.0;
mu1=0.03;
sigma1=0.02;
spotData1=equitySpotData(S0,r,d);

#Model=MertonProcess(sigma,lam,mu1,sigma1);
Model=BlackScholesProcess(sigma);

EUData=EuropeanOption(T,K)
Nsim=10000;
Nstep=30;
mc=MonteCarloConfiguration(Nsim,Nstep);

@time @show pricer(Model,spotData1,method,EUData);
@time @show pricer(Model,spotData1,mc,EUData);
@time typeof(Model)<:BlackScholesProcess ? @show(blsprice(S0,K,r,T,sigma,d)) : nothing ;
