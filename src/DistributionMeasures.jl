module DistributionMeasures

const MeasureLike = Union{AbstractMeasure,Distribution}

struct DistributionMeasure{D<:Distribution} <: AbstractMeasure
    d::D
end

end
