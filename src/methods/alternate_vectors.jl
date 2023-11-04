
struct AlternateVector{T} <: AbstractArray{T, 1}
    value_odd::T
    value_even::T
    n::Int64
end

### Implementation of the array interface
Base.size(A::AlternateVector) = (A.n,)

function Base.getindex(x::AlternateVector, ind::Int)
    @boundscheck (1 <= ind <= x.n) || throw(BoundsError(x, ind))
    ifelse(isodd(ind), x.value_odd, x.value_even)
end

# AlternateVector is closed under getindex.
function Base.getindex(A::AlternateVector, el::AbstractRange{T}) where {T <: Int}
    first_idx = el.start
    @boundscheck (1 <= first_idx <= A.n) || throw(BoundsError(A, first_idx))
    @boundscheck (1 <= el.stop <= A.n) || throw(BoundsError(A, el.stop))
    @views @inbounds odd_value = A[first_idx]
    @views @inbounds even_value = A[first_idx+step(el)]
    new_len = length(el)
    return AlternateVector(odd_value, even_value, new_len)
end

# IO
Base.showarg(io::IO, A::AlternateVector, _) = print(io, typeof(A))

# Broacasting relation against other arrays
Base.BroadcastStyle(::Type{<:AlternateVector{T}}) where {T} = Broadcast.ArrayStyle{AlternateVector{T}}()
Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternateVector{T}}, ::Broadcast.DefaultArrayStyle{0}) where {T} = a
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternateVector{T}}, ::Base.Broadcast.Style{Tuple}) where {T} = Broadcast.DefaultArrayStyle{1}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternateVector{T}}, a::Broadcast.DefaultArrayStyle{N}) where {T, N} = a

#Broacasting over AlternateVector
flatten_even(x) = x
flatten_even(x::Base.RefValue) = x.x
flatten_odd(x) = flatten_even(x)
flatten_even(x::AlternateVector) = x.value_even
flatten_odd(x::AlternateVector) = x.value_odd

function Base.materialize(bc1::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{AlternateVector{T}}, Nothing, <:F, <:R}) where {T, F, R}
    bc = Broadcast.flatten(bc1)
    func = bc.f
    args = bc.args
    axes_result = Broadcast.combine_axes(args...)
    odd_part = func(flatten_odd.(args)...)
    even_part = func(flatten_even.(args)...)
    return AlternateVector(odd_part, even_part, length(first(axes_result)))
end

function Base.sum(x::AlternateVector)
    isfinalodd = isodd(x.n)
    nhalf = div(x.n, 2)
    return (nhalf + isfinalodd) * x.value_odd + nhalf * x.value_even
end

using ChainRulesCore
function ChainRulesCore.rrule(::Type{AlternateVector}, value_odd::T, value_even::T, n::Int64) where {T}
    function AlternateVector_pb(Δapv)
        odd_v = AlternateVector(one(T), zero(T), n)
        odd_der = sum(odd_v .* Δapv)
        even_der = sum(Δapv) - odd_der
        NoTangent(), odd_der, even_der, NoTangent()
    end
    return AlternateVector(value_odd, value_even, n), AlternateVector_pb
end
