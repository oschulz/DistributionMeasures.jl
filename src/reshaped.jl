# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).


MeasureBase.effndof(d::ReshapedDistribution) = MeasureBase.effndof(d.dist)

MeasureBase.vartransform_origin(d::ReshapedDistribution) = d.dist

MeasureBase.to_origin(src::ReshapedDistribution, x) = reshape(x, size(src.dist))

MeasureBase.from_origin(trg::ReshapedDistribution, x) = reshape(x, trg.dims)
