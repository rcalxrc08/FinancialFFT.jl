using DualNumbers, Random
include("fft.jl")
a=dual(0.2,1.0);
function trial_(a)
	Random.seed!(0)
	x=(randn(10).+1im*randn(10)).*(exp(a)+a*a+a-sin(a));
	return x
end

epsilon.(fft(trial_(a)))
(fft(trial_(0.2+1e-6))-fft(trial_(0.2)))./1e-6