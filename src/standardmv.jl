# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).


@inline function MeasureBase.vartransform(trg::StandardDist{Uniform,N}, src::StandardDist{Normal,N}, x)
    _check_arraylike_match(trg, src, x)
    StatsFuns.normcdf.(x)
end


@inline function MeasureBase.vartransform(trg::StandardDist{Uniform,N}, src::StandardDist{Normal,N}, x)
    _check_arraylike_match(trg, src, x)
    StatsFuns.norminvcdf.(x)
end


MeasureBase.effndof(d::MvNormal) = length(d)

MeasureBase.vartransform_origin(mu::MvNormal) = StandardDist{Normal,1}(length(mu))

function MeasureBase.to_origin(src::MvNormal, x)
    check_varshape(src, x)
    A = cholesky(src.Σ).U
    transpose(A) \ (x - src.μ)
end

function MeasureBase.from_origin(trg::MvNormal, x)
    check_varshape(trg, x)
    A = cholesky(trg.Σ).U
    transpose(A) * x + trg.μ
end
