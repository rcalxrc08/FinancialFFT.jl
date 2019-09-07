using FinancialToolbox,DualNumbers,FinancialFFT,FinancialMonteCarlo


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
EUData=Array{FinancialMonteCarlo.AbstractPayoff}(undef,10)
EUData[1:3]=[EuropeanOption(T,K) for i in 1:3];
EUData[8:10]=[EuropeanOption(T,K) for i in 1:3];
EUData[4:7]=[AsianFloatingStrikeOption(30) for i in 4:7];
Nsim=10000;
Nstep=30;
mc=MonteCarloConfiguration(Nsim,Nstep);

@time @show pricer(Model,spotData1,method,EUData);
@time @show pricer(Model,spotData1,mc,EUData);
typeof(Model)<:BlackScholesProcess ? @time(@show(blsprice(S0,K,r,T,sigma,d))) : nothing ;
