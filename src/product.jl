# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

const _StdPowMeasure1 = PowerMeasure{<:StdMeasure,<:NTuple{1,Base.OneTo}}
const _UniformProductDist1 = Distributions.Product{Continuous,<:Uniform,<:AbstractVector{<:Uniform}}


MeasureBase.getdof(d::_UniformProductDist1) = length(d)


function _product_dist_trafo_impl(νs, μs, x)
    fwddiff(transport_def).(νs, μs, x)
end

function MeasureBase.transport_def(ν::_StdPowMeasure1, μ::_UniformProductDist1, x)
    _product_dist_trafo_impl((ν.parent,), μ.v, x)
end

function MeasureBase.transport_def(ν::_UniformProductDist1, μ::_StdPowMeasure1, x)
    _product_dist_trafo_impl(ν.v, (μ.parent,), x)
end
