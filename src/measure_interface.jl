# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

@inline MeasureBase.logdensity_def(d::Distribution, x) = DensityInterface.logdensityof(d, x)
@inline MeasureBase.unsafe_logdensityof(d::Distribution, x) = DensityInterface.logdensityof(d, x)

@inline MeasureBase.insupport(d::Distribution, x) = Distributions.insupport(d, x)

@inline MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate{0},<:Continuous}) = Lebesgue()
@inline MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate,<:Continuous}) = Lebesgue()^size(d)
@inline MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate{0},<:Discrete}) = Counting()
@inline MeasureBase.basemeasure(d::Distribution{<:ArrayLikeVariate,<:Discrete}) = Counting()^size(d)

@inline MeasureBase.insupport(d::Distribution, x) = Distributions.insupport(d, x)
@inline MeasureBase.paramnames(d::Distribution) = propertynames(d)
@inline MeasureBase.params(d::Distribution) = NamedTuple{MeasureBase.paramnames(d)}(Distributions.params(d))

