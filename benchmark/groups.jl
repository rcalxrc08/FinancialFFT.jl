#Number Type (just sigma tested) ---> Model type ----> Mode ---> zero rate type

using BenchmarkTools, DualNumbers, HyperDualNumbers, FinancialMonteCarlo, FinancialFFT, JLD2

rebase = true;
retune = false
sigma_dual = dual(0.2, 1.0);
sigma_hyper = hyper(0.2, 1.0, 1.0, 0.0);
sigma_no_dual = 0.2;

suite_num = BenchmarkGroup()

und = Underlying(100.0, 0.01);
p = 0.3;
lam = 5.0;
lamp = 30.0;
lamm = 20.0;
mu1 = 0.03;
sigma1 = 0.02;

sigma_zero = 0.2;
kappa = 0.01;
theta = 0.03;
lambda = 0.01;
rho = 0.0;

theta1 = 0.01;
k1 = 0.03;
sigma1 = 0.02;

T = 1.0;
K = 100.0;

Nsim = 10000;
Nstep = 30;

A = 600.0;
N = 18;

method_cm = CarrMadanMethod(A, N);
method_lewis = LewisMethod(700.0, 200000);
method_cm_lewis = CarrMadanLewisMethod(A, N);

r = 0.02;
r_prev = 0.019999;
lam_vec = Float64[0.999999999];
zr_scalar = ZeroRate(r);
# zr_imp = FinancialMonteCarlo.ImpliedZeroRate([r_prev, r], 2.0);

opt_0 = EuropeanOption(T, K);
opt_1 = EuropeanOption(T, 105.0);
opt_2 = EuropeanOption(T, K, false);

bs(sigma) = BlackScholesProcess(sigma, und)
kou(sigma) = KouProcess(sigma, lam, p, lamp, lamm, und);
vg(sigma) = VarianceGammaProcess(sigma, theta1, k1, und)
merton(sigma) = MertonProcess(sigma, lambda, mu1, sigma1, und)
nig(sigma) = NormalInverseGaussianProcess(sigma, theta1, k1, und);

for sigma in Number[sigma_no_dual, sigma_dual, sigma_hyper]
    for model_ in [bs(sigma), kou(sigma), vg(sigma), merton(sigma), nig(sigma)]
        for mode_ in [method_cm, method_lewis, method_cm_lewis]
            for zero_ in [zr_scalar]
                opts = [opt_0, opt_1, opt_2]
                for opt_ in opts
                    suite_num[string(typeof(sigma)), string(typeof(model_).name), string(typeof(mode_)), string(typeof(zero_)), string(opt_)] = @benchmarkable pricer($model_, $zero_, $mode_, $opt_)
                end
                suite_num[string(typeof(sigma)), string(typeof(model_).name), string(typeof(mode_)), string(typeof(zero_)), "vec"] = @benchmarkable pricer($model_, $zero_, $mode_, $opts)
            end
        end
    end
end
path_tune_params = joinpath(@__DIR__, "params.json")
if retune
    tune!(suite_num)
    BenchmarkTools.save(path_tune_params, params(suite_num))
end
loadparams!(suite_num, BenchmarkTools.load(path_tune_params)[1], :evals, :samples);
results = run(suite_num, verbose = true, seconds = 1)
median_current = median(results);
path_median_old = joinpath(@__DIR__, "median_old.jld2")
if rebase
    save_object(path_median_old, median_current)
end
median_old = load(path_median_old)["single_stored_object"]
judgement_ = judge(median_current, median_old)
[println(judgement_el.first, "    ", judgement_el.second) for judgement_el in judgement_];