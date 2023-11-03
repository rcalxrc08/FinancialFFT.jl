using FinancialToolbox, HyperDualNumbers, DualNumbers, FinancialFFT, FinancialMonteCarlo, Zygote, BenchmarkTools, FiniteDiff

A = 600.0;
const N_lewis = 20000;

const method_l = LewisMethod(A, N_lewis);
S0 = 100.0;
# S0 = hyper(100.0, 1.0, 1.0, 0.0);
K = 103.0;
# r = hyper(0.02, 1.0, 1.0, 0.0);
r = 0.02;
T = 1.0;
d = 0.0;
#sigma=dual(0.2,1.0);
sigma = 0.2;
# sigma = hyper(0.2, 1.0, 1.0, 0.0);
mu1 = 0.03;
sigma1 = 0.02;
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
zero_rate = ZeroRate(r);

Model = MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d));
# Model = KouProcess(sigma, lam, p, lamp, lamm, Underlying(S0, d));
#Model = BlackScholesProcess(sigma, Underlying(S0, d));
# Model = VarianceGammaProcess(sigma, 0.01, 0.01, Underlying(S0, d));
#Model = NormalInverseGaussianProcess(sigma, 0.01, 0.01, Underlying(S0, d));
f(sigma, lam, mu1, sigma1, S0, d, r, T, K, A) = pricer(MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d)), ZeroRate(r), LewisMethod(A, N_lewis), EuropeanOption(T, K));
# pricer(MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d)), ZeroRate(r), $method, EuropeanOption(T, K)
@btime f($sigma, $lam, $mu1, $sigma1, $S0, $d, $r, $T, $K, $A)
@btime Zygote.gradient(f, $sigma, $lam, $mu1, $sigma1, $S0, $d, $r, $T, $K, $A);

function merton_pricer_to_vol(sigma, lam, mu1, sigma1, S0, d, r, T, K, A)
    price = pricer(MertonProcess(sigma, lam, mu1, sigma1, Underlying(S0, d)), ZeroRate(r), LewisMethod(A, N_lewis), EuropeanOption(T, K))
    vol = blsimpv(S0, K, r, T, price, d)
    return vol
end
function merton_pricer_to_vol(x)
    return merton_pricer_to_vol(x...)
end
# function merton_pricer(x)
#     @views price = pricer(MertonProcess(x[1], x[2], x[3], x[4], Underlying(x[5], x[6])), ZeroRate(x[7]), method_l, EuropeanOption(x[8], x[9]))
#     return price
# end

# @show Zygote.gradient(merton_pricer, inputs_)
@btime Zygote.gradient(merton_pricer_to_vol, $sigma, $lam, $mu1, $sigma1, $S0, $d, $r, $T, $K, $A);
@show Zygote.gradient(merton_pricer_to_vol, sigma, lam, mu1, sigma1, S0, d, r, T, K, A)
@show FiniteDiff.finite_difference_gradient(merton_pricer_to_vol, [sigma, lam, mu1, sigma1, S0, d, r, T, K, A])