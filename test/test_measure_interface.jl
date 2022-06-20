# This file is a part of DistributionMeasures.jl, licensed under the MIT License (MIT).

using DistributionMeasures
using Test

import Distributions
import MeasureBase

@testset "Measure interface" begin
    c0 = Distributions.Weibull(0.7, 1.3)
    c1 = Distributions.MvNormal([0.7, 0.9], [1.4 0.5; 0.5 1.1])

    d0 = Distributions.Poisson(0.7)
    d1 = Distributions.product_distribution(Distributions.Poisson.([0.7, 1.4]))

    for d in [c0, c1, d0, d1]
        x = rand(d)
        @inline @inferred(MeasureBase.logdensity_def(d, x)) == Distributions.logpdf(d, x)
        @inline @inferred(MeasureBase.unsafe_logdensityof(d, x)) == Distributions.logpdf(d, x)
    end

    @test @inferred(MeasureBase.basemeasure(c0)) == Lebesgue(ℝ)
    @test @inferred(MeasureBase.basemeasure(c1)) == Lebesgue(ℝ) ^ 2
    @test @inferred(MeasureBase.basemeasure(d0)) == Counting(ℤ)
    @test @inferred(MeasureBase.basemeasure(d1)) == Counting(ℤ) ^ 2

    @test @inferred(MeasureBase.insupport(c0, 3)) == true
    @test @inferred(MeasureBase.insupport(c0, -3)) == false
    @test @inferred(MeasureBase.insupport(c1, [0.1, 0.2])) == true
    @test @inferred(MeasureBase.insupport(d0, 3)) == true
    @test @inferred(MeasureBase.insupport(d0, 3.2)) == false
    @test @inferred(MeasureBase.insupport(d1, [1, 2])) == true
    @test @inferred(MeasureBase.insupport(d1, [1.1, 2.2])) == false

    @test MeasureBase.paramnames(c0) == (:α, :θ)
    @test @inferred(MeasureBase.params(c0)) == (α = 0.7, θ = 1.3)
end
