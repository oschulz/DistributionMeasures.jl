"""
    struct DistributionMeasure <: AbstractMeasure

Wraps a `Distributions.Distribution` as a `MeasureBase.AbstractMeasure`.

Avoid calling `DistributionMeasure(d::Distribution)` directly. Instead, use
`AbstractMeasure(d::Distribution)` to allow for specialized `Distribution`
to `AbstractMeasure` conversions.

Use `convert(Distribution, m::DistributionMeasure)` or
`Distribution(m::DistributionMeasure)` to convert back to a `Distribution`.
"""
struct DistributionMeasure{F<:VariateForm,S<:ValueSupport,D<:Distribution{F,S}} <: AbstractMeasure
    d::D
end

@inline MeasureBase.AbstractMeasure(d::Distribution) = DistributionMeasure(d)

@inline Base.convert(::Type{AbstractMeasure}, d::Distribution) = DistributionMeasure(d)

@inline Distributions.Distribution(m::DistributionMeasure) = m.distribution
@inline Distributions.Distribution{F}(m::DistributionMeasure{F}) where {F<:VariateForm} = Distribution(m)
@inline Distributions.Distribution{F,S}(m::DistributionMeasure{F,S}) where {F<:VariateForm,S<:ValueSupport} = Distribution(m)

@inline Base.convert(::Type{Distribution}, m::DistributionMeasure) = Distribution(m)
@inline Base.convert(::Type{Distribution{F}}, m::DistributionMeasure{F}) where {F<:VariateForm} = Distribution(m)
@inline Base.convert(::Type{Distribution{F,S}}, m::DistributionMeasure{F,S}) where {F<:VariateForm,S<:ValueSupport} = Distribution(m)

@inline DensityInterface.densityof(m::DistributionMeasure) = DensityInterface.densityof(m.d)
@inline DensityInterface.densityof(m::DistributionMeasure, x) = DensityInterface.densityof(m.d, x)
@inline DensityInterface.logdensityof(m::DistributionMeasure) = DensityInterface.logdensityof(m.d)
@inline DensityInterface.logdensityof(m::DistributionMeasure, x) = DensityInterface.logdensityof(m.d, x)

@inline MeasureBase.logdensity_def(m::DistributionMeasure, x) = DensityInterface.logdensityof(m.d, x)
@inline MeasureBase.unsafe_logdensityof(m::DistributionMeasure, x) = DensityInterface.logdensityof(m.d, x)

@inline MeasureBase.insupport(m::DistributionMeasure, x) = Distributions.insupport(m.x)

@inline MeasureBase.basemeasure(m::DistributionMeasure{<:ArrayLikeVariate{0},<:Continuous}) = Lebesgue()
@inline MeasureBase.basemeasure(m::DistributionMeasure{<:ArrayLikeVariate,<:Continuous}) = Lebesgue()^size(m.d)
@inline MeasureBase.basemeasure(m::DistributionMeasure{<:ArrayLikeVariate{0},<:Discrete}) = Counting()
@inline MeasureBase.basemeasure(m::DistributionMeasure{<:ArrayLikeVariate,<:Discrete}) = Counting()^size(m.d)

@inline MeasureBase.rootmeasure(m::DistributionMeasure) = MeasureBase.basemeasure(m)


Base.rand(rng::AbstractRNG, ::Type{T}, m::DistributionMeasure) where {T<:Real} = _convert_numtype(T, rand(m.d))

function _flat_powrand(rng::AbstractRNG, ::Type{T}, d::Distribution{<:ArrayLikeVariate{0}}, sz::Dims) where {T<:Real}
    _convert_numtype(T, reshape(rand(d, prod(sz)), sz...))
end

function _flat_powrand(rng::AbstractRNG, ::Type{T}, d::Distribution{<:ArrayLikeVariate{1}}, sz::Dims) where {T<:Real}
    _convert_numtype(T, reshape(rand(d, prod(sz)), size(d)..., sz...))
end

function _flat_powrand(rng::AbstractRNG, ::Type{T}, d::ReshapedDistribution{N,<:Any,<:Distribution{<:ArrayLikeVariate{1}}}, sz::Dims) where {T<:Real,N}
    _convert_numtype(T, reshape(rand(d.dist, prod(sz)), d.dims..., sz...))
end

function _flat_powrand(rng::AbstractRNG, ::Type{T}, d::Distribution, sz::Dims) where {T<:Real,N}
    flatview(ArrayOfSimilarArrays(_convert_numtype(T, rand(d, sz))))
end

function Base.rand(rng::AbstractRNG, ::Type{T}, m::PowerMeasure{<:DistributionMeasure{<:ArrayLikeVariate{0}}, NTuple{N,Base.OneTo{Int}}}) where {T<:Real,N}
    _flat_powrand(rng, T, m.parent.d, map(length, m.axes))
end

function Base.rand(rng::AbstractRNG, ::Type{T}, m::PowerMeasure{<:DistributionMeasure{<:ArrayLikeVariate{M}}, NTuple{N,Base.OneTo{Int}}}) where {T<:Real,M,N}
    flat_data = _flat_powrand(rng, T, m.parent.d, map(length, m.axes))
    ArrayOfSimilarArrays{T,M,N}(flat_data)
end


@inline MeasureBase.paramnames(m::DistributionMeasure) = propertynames(m.d)
@inline MeasureBase.params(m::DistributionMeasure) = NamedTuple{MeasureBase.paramnames(m.d)}(Distributions.params(m.d))
