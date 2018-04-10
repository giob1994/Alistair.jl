# -----------------------------------------------
# Minimal Distributions routines
# -----------------------------------------------

# TEST:
# using Rmath

#=
macro QD2(A, B) quote
    esc(:(sum((A - B).^2)))
end
=#

abstract type AbstractDistributionType end

# #########   NORMAL DISTRIBUTION   #########

abstract type Normal<:AbstractDistributionType end

function PDF{T<:Number}(x::T, dtype::Type{Normal})
    return dnorm(x, 0, 1)
end

function PDF{Tx<:Number, Tmu<:Number, Tsig<:Number}(x::Tx,  mu::Tmu, sigma²::Tsig, dtype::Type{Normal})
    return dnorm(x, mu, sqrt(sigma²))
end

function PDF{Tx<:Number, Tmu<:Number, Tsig<:Number}(X::Array{Tx,1}, mu::Tmu, sigma²::Tsig, dtype::Type{Normal})
    # return exp.(-(X - mu).^2 / (2*sigma²)) / sqrt(2*pi*sigma²);
    return [ dnorm(x, mu, sqrt(sigma²)) for x in X ]
end

function PDF{T<:Number}(X::Array{T}, mu::Array{T,1}, Σ::Array{T},  dtype::Type{Normal})
    m, n = size(X)
    if length(dtype.mu) == n && size(dtype.Σ)[1] == n
        return sqrt(2*pi*det(Σ)) * exp.(-0.5 * (X' - row(mu)') * inv(Σ) * (X - row(mu)));
    else
        error("X, mu and Σ have not valid dimensions")
    end
end

function CDF{T<:Number}(x::T, dtype::Type{Normal})
    return pnorm(x, 0, 1)
end

function CDF{Tx<:Number, Tmu<:Number, Tsig<:Number}(x::Tx,  mu::Tmu, sigma²::Tsig, dtype::Type{Normal})
    # return CDF((sqrt(sigma²)*x + mu), Normal)
    return pnorm(x, mu, sqrt(sigma²))
end

function CDF{Tx<:Number, Tmu<:Number, Tsig<:Number}(X::Array{Tx,1}, mu::Tmu, sigma²::Tsig, dtype::Type{Normal})
    # return exp.(-(X - mu).^2 / (2*sigma²)) / sqrt(2*pi*sigma²);
    return [ pnorm(x, mu, sqrt(sigma²)) for x in X ]
end

#=
function CDF{T<:Number}(X::Array{T}, mu::Array{T,1}, Σ::Array{T},  dtype::Normal)
    return 0;
end
=#

function logLike{T<:Number}(X::Array{T,1}, mu::T, sigma²::T, dtype::Type{Normal})
    N = length(X)
    return - N/2*log(sigma²) - (0.5/sigma²)*sum((X - mu).^2)
end

function logLike{T<:Number}(X::Array{T,1}, mu::Array{T,1}, Σ::Array{T}, dtype::Type{Normal})
    N = length(X)
    return - N/2*log(det(Σ)) - 0.5 * sum(X' - row(mu)') * inv(Σ) * (X - row(mu));
end

# println(@time CDF([-1.0, 1], 1.0, 4.0, Normal))

# #########   STUDENT DISTRIBUTION   #########

abstract type Student<:AbstractDistributionType end

function PDF{Tx<:Number, Tnu<:Number}(x::Tx, ν::Tnu, dtype::Type{Student})
    # dt(x, ν) == gamma((ν+1)/2) / (sqrt(ν*pi) * gamma(ν/2)) * (1 + x.^2/ν).^(-(ν+1)/2)
    return dt(x, ν)
end

function PDF{Tx<:Number, Tnu<:Number}(X::Array{Tx,1}, ν::Tnu, dtype::Type{Student})
    return [ dt(x, ν) for x in X ]
end

function CDF{Tx<:Number, Tnu<:Number}(x::Tx, ν::Tnu, dtype::Type{Student})
    return pt(x, ν)
end

function CDF{Tx<:Number, Tnu<:Number}(X::Array{Tx,1}, ν::Tnu, dtype::Type{Student})
    return [ pt(x, ν) for x in X ]
end

# println(@time CDF(1.0, 1.0, Student))

# #########   LOGISTIC   #########

abstract type Logistic<:AbstractDistributionType end

logistic{T<:Number}(X::Array{T,1}) = 1 ./ (1 + exp.(X));

function logLike{T<:Number}(X::Array{T}, Y::Array{T}, theta::Array{T}, dtype::Type{Logistic}) 
    N = length(Y)
    return sum((logistic(X * theta)).^Y .* (1 - logistic(X * theta)).^(1 - Y)) / N
end