using FinancialToolbox,DualNumbers,FinancialFFT,FinancialMonteCarlo


A=600.0;
N=16;

method=CarrMadanMethod(A,N);
S0=100.0;
K=100.0;
r=dual(0.02,1.0);
#r=0.02;
T=1.0;
d=0.0;
#sigma=dual(0.2,1.0);
sigma=0.2;
lam=15.0;
mu1=0.03;
sigma1=0.02;
spotData1=ZeroRate(r);

#Model=MertonProcess(sigma,lam,mu1,sigma1);
Model=BlackScholesProcess(sigma,Underlying(S0,d));

EUData=EuropeanOption(T,K)
Nsim=10000;
Nstep=30;
mc=MonteCarloConfiguration(Nsim,Nstep);

@time @show pricer(Model,spotData1,method,EUData);
@time @show pricer(Model,spotData1,mc,EUData);
typeof(Model)<:BlackScholesProcess ? @time(@show(blsprice(S0,K,r,T,sigma,d))) : nothing ;
