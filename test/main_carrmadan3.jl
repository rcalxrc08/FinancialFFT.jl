using FinancialToolbox, FinancialFFT, FinancialMonteCarlo, BenchmarkTools, Zygote

A = 600.0;
N = 14;

const method_ll = CarrMadanMethod(A, N);
S0 = 100.0;
K = 100.0;
r = 0.02;
#r=0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

#Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
#Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
# EUData = EuropeanOption(T, K)
Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
# Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
#Model = NormalInverseGaussianProcess(sigma, 0.01, 0.01, Underlying(S0, d));
f(x) = @views pricer(MertonProcess(x[1], x[2], x[3], x[4], Underlying(x[5], x[6])), ZeroRate(x[7]), method_ll, EuropeanOption(x[8], x[9]));
# pricer(MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d)), ZeroRate(r), $method, EuropeanOption(T, K)
const inputs_ = [sigma, lam, mu1, sigma1, S0, d, r, T, K]
@btime Zygote.gradient(f, $inputs_);

function merton_pricer_to_vol(x)
    @views price = pricer(MertonProcess(x[1], x[2], x[3], x[4], Underlying(x[5], x[6])), ZeroRate(x[7]), method_ll, EuropeanOption(x[8], x[9]))
    @views vol = blsimpv(x[5], x[9], x[7], x[8], price, x[6])
    return vol
end

# function merton_pricer(x)
#     @views price = pricer(MertonProcess(x[1], x[2], x[3], x[4], Underlying(x[5], x[6])), ZeroRate(x[7]), method_l, EuropeanOption(x[8], x[9]))
#     return price
# end

# @show Zygote.gradient(merton_pricer, inputs_)
@show Zygote.gradient(merton_pricer_to_vol, inputs_)
