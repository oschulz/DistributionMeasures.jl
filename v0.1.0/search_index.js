var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DistributionMeasures","category":"page"},{"location":"#DistributionMeasures","page":"Home","title":"DistributionMeasures","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DistributionMeasures.","category":"page"},{"location":"","page":"Home","title":"Home","text":"DistributionMeasures provides conversions between Distributions.jl distributions and MeasureBase.jl/MeasureTheory.jl measures.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DistributionMeasures]","category":"page"},{"location":"#DistributionMeasures.DistributionMeasure","page":"Home","title":"DistributionMeasures.DistributionMeasure","text":"struct DistributionMeasure <: AbstractMeasure\n\nWraps a Distributions.Distribution as a MeasureBase.AbstractMeasure.\n\nAvoid calling DistributionMeasure(d::Distribution) directly. Instead, use AbstractMeasure(d::Distribution) to allow for specialized Distribution to AbstractMeasure conversions.\n\nUse convert(Distribution, m::DistributionMeasure) or Distribution(m::DistributionMeasure) to convert back to a Distribution.\n\n\n\n\n\n","category":"type"}]
}