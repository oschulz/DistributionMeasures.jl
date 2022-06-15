# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).


"""
    DistributionMeasures.require_insupport(μ, x)::Nothing

Checks if `x` is in the support of distribution/measure `μ`, throws an
`ArgumentError` if not.
"""
function require_insupport end

_check_insupport_pullback(ΔΩ) = NoTangent(), ZeroTangent()
function ChainRulesCore.rrule(::typeof(require_insupport), μ, x)
    return require_insupport(μ, x), _check_insupport_pullback
end

function require_insupport(μ, x::AbstractArray{T,N}) where {T,N}
    if !insupport(μ, x)
        throw(ArgumentError("x is not within the support of μ"))
    end
    return nothing
end
