# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using VariateTransformations
using Test

using Random, Statistics, LinearAlgebra
using Distributions, PDMats
using StableRNGs
using FillArrays
using ForwardDiff


@testset "standard_normal" begin
    stblrng() = StableRNG(789990641)

    @testset "StandardUvUniform" begin
        @test @inferred(VariateTransformations.StandardUvUniform()) isa VariateTransformations.StandardUvUniform

        @test @inferred(Uniform(VariateTransformations.StandardUvUniform())) isa Uniform{Float64}
        @test @inferred(Uniform(VariateTransformations.StandardUvUniform())) == Uniform()
        @test @inferred(convert(Uniform, VariateTransformations.StandardUvUniform())) == Uniform()

        d = VariateTransformations.StandardUvUniform()
        dref = Uniform()

        @test @inferred(minimum(d)) == minimum(dref)
        @test @inferred(maximum(d)) == maximum(dref)
        
        @test @inferred(params(d)) == params(dref)
        @test @inferred(partype(d)) == partype(dref)
        
        @test @inferred(location(d)) == location(dref)
        @test @inferred(scale(d)) == scale(dref)
        
        @test @inferred(eltype(typeof(d))) == eltype(typeof(dref))
        @test @inferred(eltype(d)) == eltype(dref)

        @test @inferred(length(d)) == length(dref)
        @test @inferred(size(d)) == size(dref)

        @test @inferred(mean(d)) == mean(dref)
        @test @inferred(median(d)) == median(dref)
        @test @inferred(mode(d)) == mode(dref)
        @test @inferred(modes(d)) ≈ modes(dref)
        
        @test @inferred(var(d)) == var(dref)
        @test @inferred(std(d)) == std(dref)
        @test @inferred(skewness(d)) == skewness(dref)
        @test @inferred(kurtosis(d)) == kurtosis(dref)
        
        @test @inferred(entropy(d)) == entropy(dref)
        
        for x in [-0.5, 0.0, 0.25, 0.75, 1.0, 1.5]
            @test @inferred(logpdf(d, x)) == logpdf(dref, x)
            @test @inferred(pdf(d, x)) == pdf(dref, x)
            @test @inferred(logcdf(d, x)) == logcdf(dref, x)
            @test @inferred(cdf(d, x)) == cdf(dref, x)
            @test @inferred(logccdf(d, x)) == logccdf(dref, x)
            @test @inferred(ccdf(d, x)) == ccdf(dref, x)
        end

        for p in [0.0, 0.25, 0.75, 1.0]
            @test @inferred(quantile(d, p)) == quantile(dref, p)
            @test @inferred(cquantile(d, p)) == cquantile(dref, p)
        end

        for t in [-3, 0, 3]
            @test @inferred(mgf(d, t)) == mgf(dref, t)
            @test @inferred(cf(d, t)) == cf(dref, t)
        end

        @test @inferred(rand(stblrng(), d)) == rand(stblrng(), d)
        @test @inferred(rand(stblrng(), d, 5)) == rand(stblrng(), d, 5)

        @test @inferred(truncated(VariateTransformations.StandardUvUniform(), -0.5f0, 0.7f0)) isa Uniform{Float64}
        @test truncated(VariateTransformations.StandardUvUniform(), -0.5f0, 0.7f0) == Uniform(0.0f0, 0.7f0)
        @test truncated(VariateTransformations.StandardUvUniform(), 0.2f0, 0.7f0) == Uniform(0.2f0, 0.7f0)

        @test @inferred(product_distribution(fill(VariateTransformations.StandardUvUniform(), 3))) isa VariateTransformations.StandardDist{Uniform,1}
        @test product_distribution(fill(VariateTransformations.StandardUvUniform(), 3)) == VariateTransformations.StandardDist{Uniform,1}(3)
    end


    @testset "StandardDist{Uniform,1}" begin
        @test @inferred(VariateTransformations.StandardDist{Uniform,1}(3)) isa VariateTransformations.StandardDist{Uniform,1}
        
        @test @inferred(Distributions.Product(VariateTransformations.StandardDist{Uniform,1}(3))) isa Distributions.Product{Continuous,VariateTransformations.StandardUvUniform}
        @test @inferred(convert(Product, VariateTransformations.StandardDist{Uniform,1}(3))) == Distributions.Product(Fill(VariateTransformations.StandardUvUniform(), 3))

        d = VariateTransformations.StandardDist{Uniform,1}(3)
        dref = product_distribution(fill(Uniform(), 3))

        @test @inferred(eltype(typeof(d))) == eltype(typeof(dref))
        @test @inferred(eltype(d)) == eltype(dref)

        @test @inferred(length(d)) == length(dref)
        @test @inferred(size(d)) == size(dref)

        @test @inferred(view(VariateTransformations.StandardDist{Uniform,1}(7), 3)) isa VariateTransformations.StandardUvUniform
        @test_throws BoundsError view(VariateTransformations.StandardDist{Uniform,1}(7), 9)
        @test @inferred(view(VariateTransformations.StandardDist{Uniform,1}(7), 2:4)) isa VariateTransformations.StandardDist{Uniform,1}
        @test view(VariateTransformations.StandardDist{Uniform,1}(7), 2:4) == VariateTransformations.StandardDist{Uniform,1}(3)
        @test_throws BoundsError view(VariateTransformations.StandardDist{Uniform,1}(7), 2:8)
        
        @test @inferred(mean(d)) == mean(dref)
        @test @inferred(var(d)) == var(dref)
        @test @inferred(cov(d)) == cov(dref)
        
        @test @inferred(mode(d)) == [0.5, 0.5, 0.5]
        @test @inferred(modes(d)) == Vector{Float64}[]
        
        @test @inferred(invcov(d)) == inv(cov(dref))
        @test @inferred(logdetcov(d)) == logdet(cov(dref))
        
        @test @inferred(entropy(d)) == entropy(dref)

        for x in fill.([-Inf, -1.3, 0.0, 1.3, +Inf], 3)
            @test @inferred(insupport(d, x)) == insupport(dref, x)
            @test @inferred(logpdf(d, x)) == logpdf(dref, x)
            @test @inferred(pdf(d, x)) == pdf(dref, x)
            @test @inferred(gradlogpdf(d, x)) == ForwardDiff.gradient(x -> logpdf(d, x), x)
        end
            
        @test @inferred(rand(stblrng(), d)) == rand(stblrng(), d)
        @test @inferred(rand!(stblrng(), d, zeros(3))) == rand!(stblrng(), d, zeros(3))
        @test @inferred(rand!(stblrng(), d, zeros(3, 10))) == rand!(stblrng(), d, zeros(3, 10))
    end
end