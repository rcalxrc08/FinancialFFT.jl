
struct AlternatePaddedVector{T} <: AbstractArray{T, 1}
    bound_initial_value::T
    value_even::T
    value_odd::T
    bound_final_value::T
    n::Int64
    function AlternatePaddedVector(bound_initial_value::T, value_even::T, value_odd::T, bound_final_value::T, n::Int64) where {T}
        (2 <= n) || throw("length of AlternatePaddedVector must be greater than 2. Provided is $n")
        return new{T}(bound_initial_value, value_even, value_odd, bound_final_value, n)
    end
end

### Implementation of the array interface
Base.size(A::AlternatePaddedVector) = (A.n,)

function Base.getindex(x::AlternatePaddedVector, ind::Int)
    @boundscheck (1 <= ind <= x.n) || throw(BoundsError(x, ind))
    ifelse(ind == 1, x.bound_initial_value, ifelse(ind == x.n, x.bound_final_value, ifelse(isodd(ind), x.value_odd, x.value_even)))
end

# AlternatePaddedVector is closed under getindex.
function Base.getindex(A::AlternatePaddedVector, el::AbstractRange{T}) where {T <: Int}
    n = length(el)
    (2 <= n) || throw("Trying to getindex with an AbstractRange of length $n. Provided length must be non smaller than 2.")
    first_idx = el.start
    @boundscheck (1 <= first_idx <= A.n) || throw(BoundsError(A, first_idx))
    @boundscheck (1 <= el.stop <= A.n) || throw(BoundsError(A, el.stop))
    @views @inbounds bound_initial_value = A[first_idx]
    step_el = step(el)
    next_idx = first_idx + step_el
    @views @inbounds even_value = A[next_idx] #This is a trick, in case n<4 it's breaking the bounds, but it's fine the function is well behaving anyway
    @views @inbounds odd_value = A[next_idx+step_el] #This is a trick, in case n<4 it's breaking the bounds, but it's fine the function is well behaving anyway
    @views @inbounds bound_final_value = A[last(el)]
    return AlternatePaddedVector(bound_initial_value, even_value, odd_value, bound_final_value, n)
end

# IO
Base.showarg(io::IO, A::AlternatePaddedVector, _) = print(io, typeof(A))

# Broacasting relation against other arrays
Base.BroadcastStyle(::Type{<:AlternatePaddedVector{T}}) where {T} = Broadcast.ArrayStyle{AlternatePaddedVector{T}}()
# Relation against scalars
Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, ::Broadcast.DefaultArrayStyle{0}) where {T} = a
# Relation against tuples
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, ::Base.Broadcast.Style{Tuple}) where {T} = Broadcast.DefaultArrayStyle{1}()
#Relation against general arrays
Base.BroadcastStyle(::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, a::Broadcast.DefaultArrayStyle{N}) where {T, N} = a
#Relation against general AlternateVectors
Base.BroadcastStyle(a::Broadcast.ArrayStyle{AlternatePaddedVector{T}}, ::Broadcast.ArrayStyle{AlternateVector{T}}) where {T} = a

#Broacasting over AlternatePaddedVector
apv_flatten_even(x) = x
apv_flatten_even(x::Base.RefValue) = x.x
apv_flatten_initial(x) = apv_flatten_even(x)
apv_flatten_odd(x) = apv_flatten_even(x)
apv_flatten_final(x) = apv_flatten_even(x)
apv_flatten_initial(x::AlternatePaddedVector) = x.bound_initial_value
apv_flatten_even(x::AlternatePaddedVector) = x.value_even
apv_flatten_odd(x::AlternatePaddedVector) = x.value_odd
apv_flatten_initial(x::AlternateVector) = x.value_odd
apv_flatten_odd(x::AlternateVector) = x.value_odd
apv_flatten_even(x::AlternateVector) = x.value_even
apv_flatten_final(x::AlternateVector) = @views x[end]
apv_flatten_final(x::AlternatePaddedVector) = x.bound_final_value

function Base.materialize(bc1::Base.Broadcast.Broadcasted{Base.Broadcast.ArrayStyle{AlternatePaddedVector{T}}, Nothing, <:F, <:R}) where {T, F, R}
    bc = Broadcast.flatten(bc1)
    func = bc.f
    args = bc.args
    axes_result = Broadcast.combine_axes(args...)
    in_part = func(apv_flatten_initial.(args)...)
    even_part = func(apv_flatten_even.(args)...)
    odd_part = func(apv_flatten_odd.(args)...)
    fin_part = func(apv_flatten_final.(args)...)
    return AlternatePaddedVector(in_part, even_part, odd_part, fin_part, length(first(axes_result)))
end

using ChainRulesCore
function ChainRulesCore.rrule(::Type{AlternatePaddedVector}, bound_initial_value::T, value_even::T, value_odd::T, bound_final_value::T, n::Int64) where {T}
    function AlternatePaddedVector_pb(Δapv)
        der_bound_initial_value = @views Δapv[1]
        odd_v = AlternatePaddedVector(zero(T), zero(T), one(T), zero(T), n)
        odd_der = sum(odd_v .* Δapv)
        even_v = AlternatePaddedVector(zero(T), one(T), zero(T), zero(T), n)
        even_der = sum(even_v .* Δapv)
        der_bound_final_value = @views Δapv[end]
        NoTangent(), der_bound_initial_value, even_der, odd_der, der_bound_final_value, NoTangent()
    end
    return AlternatePaddedVector(bound_initial_value, value_even, value_odd, bound_final_value, n), AlternatePaddedVector_pb
end
