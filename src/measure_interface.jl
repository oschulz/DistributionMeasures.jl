# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

@inline MeasureBase.logdensity_def(d::Distribution, x) = DensityInterface.logdensityof(d, x)
@inline function MeasureBase.unsafe_logdensityof(d::Distribution, x)
    DensityInterface.logdensityof(d, x)
end

@inline MeasureBase.insupport(d::Distribution, x) = Distributions.insupport(d, x)

@inline function MeasureBase.basemeasure(
    d::Distribution{<:ArrayLikeVariate{0},<:Continuous},
)
    Lebesgue()
end
@inline function MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate,<:Continuous})
    Lebesgue()^size(d)
end
@inline function MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate{0},<:Discrete})
    Counting()
end
@inline function MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate,<:Discrete})
    Counting()^size(d)
end

@inline MeasureBase.paramnames(d::Distribution) = propertynames(d)
@inline function MeasureBase.params(d::Distribution)
    NamedTuple{propertynames(d)}(Distributions.params(d))
end

@inline MeasureBase.testvalue(d::Distribution) = testvalue(basemeasure(d))

@inline function MeasureBase.basemeasure(d::Distributions.Poisson)
    Counting(MeasureBase.BoundedInts(static(0), static(Inf)))
end
@inline function MeasureBase.basemeasure(
    d::Distributions.Product{<:Any,<:Distributions.Poisson},
)
    Counting(MeasureBase.BoundedInts(static(0), static(Inf)))^size(d)
end

function MeasureBase.testvalue(::Type{T}, d::Distribution) where {T}
    rand(MeasureBase.FixedRNG(), d)
end

MeasureBase.∫(f, base::Distribution) = MeasureBase.∫(f, convert(AbstractMeasure, base))
