using DistributionMeasures
using Test

@testset "DistributionMeasures.jl" begin
    include("test_autodiff_utils.jl")
    include("test_standard_dist.jl")
    include("test_standard_uniform.jl")
    include("test_standard_normal.jl")
end
