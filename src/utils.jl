@inline _convert_numtype(::Type{T}, x::T) where {T<:Real} = x
@inline _convert_numtype(::Type{T}, x::AbstractArray{T}) where {T<:Real} = x
@inline _convert_numtype(::Type{T}, x::U) where {T<:Real,U<:Real} = T(X)
_convert_numtype(::Type{T}, x::AbstractArray{U}) where {T<:Real,U<:Real} = T.(x)
_convert_numtype(::Type{T}, x) where {T<:Real} = fmap(elem -> _convert_numtype(T, elem), x)
