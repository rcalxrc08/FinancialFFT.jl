# Scalar hooks: the kernels call these instead of the functions of Base, and the extensions of the AD back-ends add
# methods for their number types where Base does not work or is not differentiable.

real_mod(x) = real(x)
imag_mod(x) = imag(x)
# Mooncake cannot differentiate the complex sqrt of Base: its extension overlays this hook.
sqrt_mod(x) = sqrt(x)
# Base defines cis for real numbers only: the TaylorSeries extension overloads this hook.
cis_mod(x) = cis(x)
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
# Quantities that are constants for differentiation (truncation ranges, grids). ReverseDiff does not honor
# ChainRulesCore.@ignore_derivatives, so its extension strips the tracking here, and Mooncake ignores it too, so its
# extension makes this hook a primitive with zero derivative.
ignore_derivatives_mod(x) = x
