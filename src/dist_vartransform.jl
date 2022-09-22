# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

const _AnyStdUniform = Union{StandardUniform,Uniform}
const _AnyStdNormal = Union{StdNormal,Normal}

const _AnyStdDistribution = Union{_AnyStdUniform,_AnyStdNormal}

_std_measure(::Type{<:_AnyStdUniform}) = StandardUniform
_std_measure(::Type{<:_AnyStdNormal}) = StdNormal

_std_measure(::Type{M}, ::StaticInt{1}) where {M<:_AnyStdDistribution} = M()
_std_measure(::Type{M}, dof::Integer) where {M<:_AnyStdDistribution} = M(dof)
function _std_measure_for(::Type{M}, μ::Any) where {M<:_AnyStdDistribution}
    _std_measure(_std_measure(M), getdof(μ))
end

function MeasureBase.transport_to(::Type{NU}, μ) where {NU<:_AnyStdDistribution}
    transport_to(_std_measure_for(NU, μ), μ)
end
function MeasureBase.transport_to(ν, ::Type{MU}) where {MU<:_AnyStdDistribution}
    transport_to(ν, _std_measure_for(MU, ν))
end
