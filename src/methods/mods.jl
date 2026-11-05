# Scalar hooks: the kernels call these instead of the functions of Base, and the extensions of the AD back-ends add
# methods for their number types where Base does not work or is not differentiable.

# HyperDualNumbers defines neither real nor imag for its complex hyper-duals, Hyper{<:Complex}
# (JuliaDiff/HyperDualNumbers.jl#38, open since 2022): its extension overloads these two hooks, the only one that does.
# Every other back-end goes through Base. Once real and imag exist upstream, the hooks and the overloads can go.
real_mod(x) = real(x)
imag_mod(x) = imag(x)
# Mooncake cannot differentiate the complex sqrt of Base: its extension overlays this hook.
sqrt_mod(x) = sqrt(x)
# The abs of a Taylor series refuses a constant term of zero, which a density has at some nodes of its tails: the
# TaylorSeries extension overloads this hook.
abs_mod(x) = abs(x)
# DualNumbers and HyperDualNumbers have no sincos: their extensions overload this hook.
sincos_mod(x) = sincos(x)
# real(exp(z)), without the imaginary part of the exponential.
exp_mod(x) = exp(real_mod(x)) * cos(imag_mod(x))
# The value of a number without its derivatives (extensions: the value of a dual, of a tracked real, the constant term
# of a Taylor series).
v_value_mod(s::Real) = s
# Quantities that are constants for differentiation (truncation ranges, grids): the value without derivatives. Use it
# inside ChainRulesCore.@ignore_derivatives, which Zygote honors (and which keeps it from tracing the computation of the
# quantity). The value covers the dual numbers, the Taylor series and ReverseDiff, which does not honor the macro;
# Mooncake ignores the macro too, so its extension makes this hook a primitive with zero derivative.
ignore_derivatives_mod(x) = v_value_mod(x)
