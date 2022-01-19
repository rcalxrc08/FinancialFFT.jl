using DualNumbers, HyperDualNumbers

function real_mod(x::Hyper)
    return hyper(real(x.value), real(x.epsilon1), real(x.epsilon2), real(x.epsilon12))
end
real_mod(x) = real(x)

#Base.hash(x::Dual) = Base.hash(Base.hash("r") - Base.hash(x.value) - Base.hash("i") - Base.hash(x.epsilon))