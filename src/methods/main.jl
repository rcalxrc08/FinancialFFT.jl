A=400.0;
N=12;

method=CarrMadanMethod(A,N);
S0=100.0;
K=100.0;
r=0.02;
T=1.0;
d=0.01;
sigma=0.2; 
lam=5.0; 
mu1=0.03; 
sigma1=0.02;
spotData1=equitySpotData(S0,r,d);

Model=MertonProcess(sigma,lam,mu1,sigma1);

EUData=EuropeanOption(T,K)

pricer(method,Model,spotData1,EUData)