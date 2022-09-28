# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

module DistributionMeasures

using LinearAlgebra: Diagonal, dot, cholesky

import Random
using Random: AbstractRNG, rand!

import DensityInterface
using DensityInterface: logdensityof, densityof

import MeasureBase
using MeasureBase: AbstractMeasure, Lebesgue, Counting, ℝ
using MeasureBase: StdMeasure, StdUniform, StdExponential, StdLogistic, StdNormal
using MeasureBase: PowerMeasure, WeightedMeasure
using MeasureBase: basemeasure, testvalue
using MeasureBase: getdof, checked_arg
using MeasureBase: transport_to, transport_def, transport_origin, from_origin, to_origin
using MeasureBase: NoTransportOrigin, NoTransport

import Distributions
using Distributions: Distribution, VariateForm, ValueSupport, ContinuousDistribution
using Distributions: Univariate, Multivariate, ArrayLikeVariate, Continuous, Discrete
using Distributions: Uniform, Exponential, Logistic, Normal
using Distributions: MvNormal, Beta, Dirichlet
using Distributions: ReshapedDistribution

import Statistics
import StatsBase
import StatsFuns
import PDMats

using IrrationalConstants: log2π, invsqrt2π

using Static: True, False, StaticInt, static
using FillArrays: Fill, Ones, Zeros

import ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, unthunk, @thunk

import ForwardDiff
using ForwardDiffPullbacks: fwddiff

import Functors
using Functors: fmap

using ArgCheck: @argcheck

using ArraysOfArrays: ArrayOfSimilarArrays, flatview

const MeasureLike = Union{AbstractMeasure,Distribution}
export MeasureLike

include("utils.jl")
include("autodiff_utils.jl")
include("measure_interface.jl")
include("standard_dist.jl")
include("standard_uniform.jl")
include("standard_normal.jl")
include("distribution_measure.jl")
include("dist_vartransform.jl")
include("univariate.jl")
include("standardmv.jl")
include("product.jl")
include("reshaped.jl")
include("dirichlet.jl")

export DistributionMeasure
export StandardDist

end # module
