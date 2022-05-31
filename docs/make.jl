using DistributionMeasures
using Documenter

DocMeta.setdocmeta!(DistributionMeasures, :DocTestSetup, :(using DistributionMeasures); recursive=true)

makedocs(;
    modules=[DistributionMeasures],
    authors="Chad Scherrer <chad.scherrer@gmail.com> and contributors",
    repo="https://github.com/cscherrer/DistributionMeasures.jl/blob/{commit}{path}#{line}",
    sitename="DistributionMeasures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cscherrer.github.io/DistributionMeasures.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cscherrer/DistributionMeasures.jl",
    devbranch="main",
)
