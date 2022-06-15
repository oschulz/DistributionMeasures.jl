# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

@inline MeasureBase.getdof(::Distribution{Univariate}) = 1


@inline MeasureBase.check_dof(a::Distribution{Univariate}, b::Distribution{Univariate}) = nothing

@inline MeasureBase.vartransform_origin(d::Distribution{Univariate,Continuous}) = StandardUniform()


# Use ForwardDiff for univariate distribution transformations:
@inline function ChainRulesCore.rrule(::typeof(vartransform_def), trg::Distribution{Univariate}, src::Distribution{Univariate}, x::Any)
    ChainRulesCore.rrule(fwddiff(vartransform_def), trg, src, x)
end


_dist_params_numtype(d::Distribution) = promote_type(map(typeof, Distributions.params(d))...)


@inline _trafo_cdf(d::Distribution{Univariate,Continuous}, x::Real) = _trafo_cdf_impl(_dist_params_numtype(d), d, x)

@inline _trafo_cdf_impl(::Type{<:Real}, d::Distribution{Univariate,Continuous}, x::Real) = cdf(d, x)

@inline function _trafo_cdf_impl(::Type{<:Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, x::ForwardDiff.Dual{TAG}) where {N,TAG}
    x_v = ForwardDiff.value(x)
    u = cdf(d, x_v)
    dudx = pdf(d, x_v)
    ForwardDiff.Dual{TAG}(u, dudx * ForwardDiff.partials(x))
end


@inline _trafo_quantile(d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl(_dist_params_numtype(d), d, u)

@inline _trafo_quantile_impl(::Type{<:Real}, d::Distribution{Univariate,Continuous}, u::Real) = _trafo_quantile_impl_generic(d, u)

@inline function _trafo_quantile_impl(::Type{<:Union{Integer,AbstractFloat}}, d::Distribution{Univariate,Continuous}, u::ForwardDiff.Dual{TAG}) where {TAG}
    x = _trafo_quantile_impl_generic(d, ForwardDiff.value(u))
    dxdu = inv(pdf(d, x))
    ForwardDiff.Dual{TAG}(x, dxdu * ForwardDiff.partials(u))
end


@inline _trafo_quantile_impl_generic(d::Distribution{Univariate,Continuous}, u::Real) = quantile(d, u)

# Workaround for Beta dist, ForwardDiff doesn't work for parameters:
@inline _trafo_quantile_impl_generic(d::Beta{T}, u::Real) where {T<:ForwardDiff.Dual} = convert(float(typeof(u)), NaN)
# Workaround for Beta dist, current quantile implementation only supports Float64:
@inline _trafo_quantile_impl_generic(d::Beta{T}, u::Union{Integer,AbstractFloat}) where {T<:Union{Integer,AbstractFloat}} = _trafo_quantile_impl(T, d, convert(promote_type(Float64, typeof(u)), u))


#=
# ToDo:

# Workaround for rounding errors that can result in quantile values outside of support of Truncated:
@inline function _trafo_quantile_impl_generic(d::Truncated{<:Distribution{Univariate,Continuous}}, u::Real)
    x = quantile(d, u)
    T = typeof(x)
    min_x = T(minimum(d))
    max_x = T(maximum(d))
    if x < min_x && isapprox(x, min_x, atol = 4 * eps(T))
        min_x
    elseif x > max_x && isapprox(x, max_x, atol = 4 * eps(T))
        max_x
    else
        x
    end
end

# Workaround for rounding errors that can result in quantile values outside of support of Truncated:
@inline function _trafo_quantile_impl_generic(d::Truncated{<:Distribution{Univariate,Continuous}}, u::Real)
    x = quantile(d, u)
    T = typeof(x)
    min_x = T(minimum(d))
    max_x = T(maximum(d))
    if x < min_x && isapprox(x, min_x, atol = 4 * eps(T))
        min_x
    elseif x > max_x && isapprox(x, max_x, atol = 4 * eps(T))
        max_x
    else
        x
    end
end
=#


@inline function _result_numtype(d::Distribution{Univariate}, x::T) where {T<:Real}
    # float(promote_type(T, eltype(Distributions.params(d))))
    firsttype(first(typeof(x), promote_type(map(eltype, Distributions.params(d))...)))
end


@inline function MeasureBase.to_origin(src::Distribution{Univariate,Continuous}, x::Real)
    R = _result_numtype(src, x)
    if Distributions.insupport(src, x)
        y = _trafo_cdf(src, x)
        convert(R, y)
    else
        convert(R, NaN)
    end
end


@inline function MeasureBase.from_origin(trg::Distribution{Univariate,Continuous}, x::T) where {T <: Real}
    R = _result_numtype(trg, x)
    TF = float(T)
    if 0 <= x <= 1
        # Avoid x ≈ 0 and x ≈ 1 to avoid infinite variate values for target distributions with infinite support:
        mod_x = ifelse(x == 0, zero(TF) + eps(TF), ifelse(x == 1, one(TF) - eps(TF), convert(TF, x)))
        y = _trafo_quantile(trg, mod_x)
        convert(R, y)
    else
        convert(R, NaN)
    end
end


function _rescaled_to_origin(src::Distribution{Univariate}, x::T) where {T<:Real}
    src_offs, src_scale = location(src), scale(src)
    y = (x - src_offs) / src_scale
    convert(_result_numtype(src, x), y)
end

function _origin_to_rescaled(trg::Distribution{Univariate}, x::T) where {T<:Real}
    trg_offs, trg_scale = location(trg), scale(trg)
    y = muladd(x, trg_scale, trg_offs)
    convert(_result_numtype(src, x), y)
end

@inline MeasureBase.vartransform_origin(d::Uniform) = StandardUniform()
@inline MeasureBase.to_origin(src::Uniform, x::Real) = _rescaled_to_origin(src, x)
@inline MeasureBase.from_origin(trg::Uniform, x::Real) = _origin_to_rescaled(trg, x)

@inline MeasureBase.vartransform_origin(d::Normal) = StandardNormal()
@inline MeasureBase.to_origin(src::Normal, x::Real) = _rescaled_to_origin(src, x)
@inline MeasureBase.from_origin(trg::Normal, x::Real) = _origin_to_rescaled(trg, x)

@inline MeasureBase.vartransform_def(::StandardUniform, ::StandardNormal, x::Real) = StatsFuns.normcdf(x)
@inline MeasureBase.vartransform_def(::StandardNormal, ::StandardUniform, x::Real) = StatsFuns.norminvcdf(x)


function MeasureBase.vartransform_def(trg::Distribution{Univariate}, src::StandardDist{D,1}, x::AbstractVector{<:Real}) where D
    @_adignore if !(size(src) == size(x) == (1,))
        throw(ArgumentError("Length of src and length of x must be one"))
    end
    return vartransform_def(trg, StandardDist{D}(), first(x))
end

function MeasureBase.vartransform_def(trg::StandardDist{D,1}, src::Distribution{Univariate}, x::Real) where D
    @_adignore size(trg) == (1,) || throw(ArgumentError("Length of trg must be one"))
    return Fill(vartransform_def(StandardDist{D}(), src, x))
end
