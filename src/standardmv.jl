# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).


MeasureBase.getdof(ν::MvNormal) = length(ν)

MeasureBase.vartransform_origin(ν::MvNormal) = StandardDist{Normal,1}(length(ν))

function MeasureBase.to_origin(ν::MvNormal, x)
    A = cholesky(ν.Σ).L
    b = ν.μ
    A \ (x - b)
end

function MeasureBase.from_origin(ν::MvNormal, x)
    A = cholesky(ν.Σ).L
    b = ν.μ
    muladd(A, x, b)
end
