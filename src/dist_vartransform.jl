# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).


const StdDistribution = StandardDist{<:Union{Uniform,Normal},0}
const AnyStdUv = Union{StdMeasure,StdDistribution}

const StdAvMeasure{N} = PowerMeasure{<:StdMeasure,<:NTuple{N,Base.OneTo}}
const StdAvDistribution{N} = StandardDist{<:Union{Uniform,Normal},N}
const AnyStdAv{N} = Union{StdAvMeasure{N},StdAvDistribution{N}}

const StdMvMeasure = StdAvMeasure{1}
const StdMvDistribution = StdAvDistribution{1}
const AnyStdMv = AnyStdAv{1}


std_univariate(mu::AnyStdUv) = mu
std_univariate(mu::StdAvMeasure) = mu.parent
std_univariate(::StandardDist{D}) where D = D()


_matching_stddist(::Type{D}, d::Distribution{Univariate,Continuous}) where {D<:Union{StandardUniform,StandardNormal}} = StandardDist{T}()
_matching_stddist(::Type{D}, d::ContinuousDistribution) where {D<:Union{StandardUniform,StandardNormal}} = StandardDist{T}(effndof(d))

MeasureBase.vartransform(::Type{D}, ::MeasureLike) where {D<:Union{StandardUniform,StandardNormal}} = vartransform(_matching_stddist(D,d), d)
MeasureBase.vartransform(::MeasureLike, ::Type{D}) where {D<:Union{StandardUniform,StandardNormal}} = vartransform(_matching_stddist(D,d), d)


@inline MeasureBase.select_vartransform_intermediate(::AnyStdMv, mu::StdDistribution) = mu
@inline MeasureBase.select_vartransform_intermediate(nu::StdDistribution, ::AnyStdMv) = nu




"""
    DistributionMeasures.check_varshape(μ, x)::Nothing

Checks if `x` has the correct shape/size for a variate of measure-like object
`μ`, throws an `ArgumentError` if not.
"""
function check_varshape end

_check_varshape_pullback(ΔΩ) = NoTangent(), ZeroTangent()
ChainRulesCore.rrule(::typeof(check_varshape), μ, x) = check_varshape(μ, x), _check_varshape_pullback


function MeasureBase.check_varshape(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray{T,N}) where {T,N}
    dist_size = size(d)
    var_size = size(x)
    if dist_size != var_size
        throw(ArgumentError("Measure has variates of size $dist_size but given variate has size $var_size"))
    end
end


function _check_arraylike_match(trg::Distribution{<:ArrayLikeVariate{N},Continuous}, src::Distribution{<:ArrayLikeVariate{N},Continuous}, x) where N
    @_adignore begin
        @argcheck x isa AbstractArray{<:Real,N} 
        @argcheck size(trg) == size(src) == size(x)
    end
end


function MeasureBase.vartransform(trg::StandardDist{D,N}, src::StandardDist{D,N}, x) where {D,N}
    _check_arraylike_match(trg, src, x)
    return x
end

function MeasureBase.vartransform(trg_d::DT, src_d::DT, src_v) where {DT <: StdMvMeasure}
    _check_arraylike_match(trg, src, x)
    return src_v
end
