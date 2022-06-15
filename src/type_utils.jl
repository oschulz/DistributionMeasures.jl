# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

"""
    DistributionMeasures.firsttype(::Type{T}, ::Type{U}) where {T<:Real,U<:Real}

Return the first type, but as a dual number type if the second one is dual.

If `U <: ForwardDiff.Dual{tag,<:Real,N}`, returns `ForwardDiff.Dual{tag,T,N}`,
otherwise returns `T`
"""
function firsttype end

firsttype(::Type{T}, ::Type{U}) where {T<:Real,U<:Real} = T
firsttype(::Type{T}, ::Type{<:ForwardDiff.Dual{tag,<:Real,N}}) where {T<:Real,tag,N} = ForwardDiff.Dual{tag,T,N}
