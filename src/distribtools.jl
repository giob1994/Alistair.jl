# -----------------------------------------------
# Minimal Distributions routines
# -----------------------------------------------

#=
macro QD2(A, B) quote
    esc(:(sum((A - B).^2)))
end
=#

abstract type AbstractDistributionType end

# #########   NORMAL DISTRIBUTION   #########

struct Normal<:AbstractDistributionType
    mu
    sigma²

    function Normal{T<:Number}(mu::T, sigma²::T)
        new(mu, sigma²)
    end

    function Normal{T<:Number}(mu::Array{T}, sigma²::Array{T})
        new(mu, sigma²)
    end
end

function PDF{T<:Number}(X::Array{T,1}, dtype::Normal)
    return exp.(-(X - dtype.mu).^2 / (2*dtype.sigma²)) / sqrt(2*pi*dtype.sigma²);
end

function PDF{T<:Number}(X::Array{T},  dtype::Normal)
    m, n = size(X)
    if length(dtype.mu) == n && size(dtype.sigma²)[1] == n
        return sqrt(2*pi*det(dtype.sigma²)) * exp.(-0.5 * (X' - row(dtype.mu)') * inv(dtype.sigma²) * (X - row(dtype.mu)));
    else
        error("X, mu and sigma² have not valid dimensions")
    end
end

function logLike{T<:Number}(X::Array{T,1}, dtype::Normal)
    N = length(X)
    return -0.5*N*log(dtype.sigma²) - (0.5/dtype.sigma)*sum((X - dtype.mu).^2)
end

# ON HOLD
#=
function normalLogLike{T<:Number}(X::Array{T}, mu::Array{T}, sigma²::Array{T})
    N = size(X)[1]
    k = size(mu)
    returni 1
end
=#

# #########   LOGISTIC   #########


struct Logistic<:AbstractDistributionType
    theta

    function Logistic()
        new(0)
    end

    function Logistic{T<:Number}(theta::Array{T})
        new(theta)
    end
end

logistic{T<:Number}(X::Array{T,1}) = 1 ./ (1 + exp.(X));

function logLike{T<:Number}(X::Array{T}, Y::Array{T}, dtype::Logistic) 
    N = length(Y)
    return sum((logistic(X * dtype.theta)).^Y .* (1 - logistic(X * dtype.theta)).^(1 - Y)) / N
end

function logLike{T<:Number}(X::Array{T}, Y::Array{T}, theta::Array{T}, dtype::Logistic) 
    N = length(Y)
    return sum((logistic(X * theta)).^Y .* (1 - logistic(X * theta)).^(1 - Y)) / N
end