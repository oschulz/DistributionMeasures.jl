module DistributionMeasures

import Random
using Random: AbstractRNG

import DensityInterface
using DensityInterface: logdensityof

import MeasureBase
using MeasureBase: AbstractMeasure, Lebesgue, Counting
using MeasureBase: PowerMeasure

import Distributions
using Distributions: Distribution, VariateForm, ValueSupport
using Distributions: ArrayLikeVariate, Continuous, Discrete
using Distributions: ReshapedDistribution

import Functors
using Functors: fmap

using ArraysOfArrays: ArrayOfSimilarArrays, flatview


include("utils.jl")
include("distribution_measure.jl")


const MeasureLike = Union{AbstractMeasure,Distribution}

export MeasureLike, DistributionMeasure


end # module
