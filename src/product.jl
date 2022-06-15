# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

MeasureBase.getdof(d::Distributions.Product) = length(d)

function _product_dist_trafo_impl(trgs, srcs, x)
    _check_arraylike_match(trg, src, x)
    fwddiff(vartransform_def).(trgs, srcs, x)
end

function MeasureBase.vartransform_def(trg::Distributions.Product, src::Distributions.Product, x)
    _product_dist_trafo_impl(trg.v, src.v, x)
end

function MeasureBase.vartransform_def(trg::AnyStdMv, src::Distributions.Product, x)
    _product_dist_trafo_impl((std_univariate(trg),), src.v, x)
end

function MeasureBase.vartransform_def(trg::Distributions.Product, src::AnyStdMv, x)
    _product_dist_trafo_impl(trg.v, (std_univariate(src),), x)
end
