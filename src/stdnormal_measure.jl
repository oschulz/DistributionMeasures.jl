struct StdNormal <: MeasureBase.StdMeasure end

export StdNormal

@inline _isreal(x) = False
@inline _isreal(x::Real) = True

@inline MeasureBase.insupport(d::StdNormal, x) = _isreal(x)
@inline MeasureBase.insupport(d::StdNormal) = _isreal

@inline MeasureBase.logdensity_def(::StdNormal, x) = -x^2 / 2
@inline MeasureBase.basemeasure(::StdNormal) = WeightedMeasure(static(-0.5 * log2π), Lebesgue(ℝ))

@inline MeasureBase.getdof(::StdNormal) = static(1)

@inline Base.rand(rng::Random.AbstractRNG, ::Type{T}, ::StdNormal) where {T} = randn(rng, T)


@inline MeasureBase.StdMeasure(::typeof(randn)) = StdNormal()
