# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

module DistributionMeasures

using LinearAlgebra: Diagonal, dot

import Random
using Random: AbstractRNG, rand!

import DensityInterface
using DensityInterface: logdensityof

import MeasureBase
using MeasureBase: AbstractMeasure, Lebesgue, Counting
using MeasureBase: StdMeasure, StdNormal, StdUniform, StdExponential
using MeasureBase: PowerMeasure

import Distributions
using Distributions: Distribution, VariateForm, ValueSupport, ContinuousDistribution
using Distributions: Univariate, Multivariate, ArrayLikeVariate, Continuous, Discrete
using Distributions: Uniform, Normal, MvNormal
using Distributions: ReshapedDistribution

import Statistics
import StatsBase
import StatsFuns
import PDMats

using IrrationalConstants: log2π, invsqrt2π

using FillArrays: Fill, Ones, Zeros

import ChainRulesCore
using ChainRulesCore: ZeroTangent, NoTangent, unthunk, @thunk

import ForwardDiff
using ForwardDiffPullbacks: fwddiff

import Functors
using Functors: fmap

using ArraysOfArrays: ArrayOfSimilarArrays, flatview


include("utils.jl")
include("autodiff_utils.jl")
include("standard_dist.jl")
include("standard_uniform.jl")
include("standard_normal.jl")
include("distribution_measure.jl")


const MeasureLike = Union{AbstractMeasure,Distribution}

export MeasureLike, DistributionMeasure

export StandardDist


end # module
