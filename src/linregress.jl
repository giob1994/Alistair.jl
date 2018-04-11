# -----------------------------------------------
# Linear Regression routines
# -----------------------------------------------

"""
Linear regression function
"""
#=
function linregress{T<:Number}(X::Array{T}, Y::Array{T}, regtype=OLS(); fast=false)
    if typeof(regtype) == OLS 
        return ols_linfit(X, Y, regtype.intercept, regtype.robust, fast=fast)
    elseif typeof(regtype) == GLS
        return gls_linfit(X, Y, regtype.omega, regtype.intercept, regtype.robust, fast=fast)
    elseif typeof(regtype) == FGLS
        return fgls_linfit(X, Y, regtype.intercept, regtype.robust, fast=fast)
    elseif typeof(regtype) == IteratedFGLS
        return iteratedfgls_linfit(X, Y, regtype.iterations, regtype.intercept, regtype.robust, fast=fast)
    elseif false
        return false
    end
end
=#

function linregress{T<:Number}(X::Array{T}, Y::Array{T}, regtype=OLS(); fast=false)
    if isdefined(regtype, :intercept)
        X = regtype.intercept ? addintercept(X) : nointercept(X)
    end 
    return regtype(X, Y, fast=fast)   
end

# #########   OLS   #########

#=
function ols_linfit{T<:Number}(X::Array{T}, Y::Array{T}, intercept=true, robust=false; fast=false)
    X = intercept ? addintercept(X) : nointercept(X)
    ols_beta = (X' * X) \ (X' * Y)
    residuals = Y - aprod(X, ols_beta)
    if robust == false || typeof(robust) == BasicVariance
        # s2 = sum(residuals.^2) / (length(Y) - length(ols_beta))
        # variance = s2 .* inv(X' * X)
        variance = BasicVariance(X, residuals, length(ols_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return fast ? (ols_beta, variance) : linearfitresult(OLS, X, Y, ols_beta, residuals, variance, robust)  
end
=#

function (olsreg::OLS)(X::Array{T}, Y::Array{T}; fast=false) where {T<:Number}
    ols_beta = (X' * X) \ (X' * Y)
    residuals = Y - aprod(X, ols_beta)
    if olsreg.robust == false || typeof(olsreg.robust) == BasicVariance
        variance = BasicVariance(X, residuals, length(ols_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return fast ? (ols_beta, variance) : linearfitresult(OLS, X, Y, ols_beta, residuals, variance, olsreg.robust) 
end

# #########   GLS   #########

function gls_linfit{T<:Number}(X::Array{T}, Y::Array{T}, omega::Array{T}, intercept=true, robust=false; fast=false)
    X = intercept ? addintercept(X) : nointercept(X)
    # Use the supplied Ω:
    gls_beta = (X' * inv(omega) * X) \ (X' * inv(omega) * Y)
    residuals = Y - aprod(X, gls_beta)
    if robust == false || typeof(robust) == BasicVariance
        variance = BasicVariance(X, residuals, length(gls_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return fast ? (gls_beta, variance) : linearfitresult(GLS, X, Y, gls_beta, residuals, variance, robust) 
end

# #########   FGLS   #########

function fgls_linfit{T<:Number}(X::Array{T}, Y::Array{T}, intercept=true, robust=false; fast=false)
    X = intercept ? addintercept(X) : nointercept(X)
    # (1) Execute OLS regression to compute residuals:
    residuals = Y - X * ((X' * X) \ (X' * Y))
    # (2) Use residuals to estimate Ω:
    omega = diagm(squeeze(residuals, 2).^2)
    fgls_beta = (X' * inv(omega) * X) \ (X' * inv(omega) * Y)
    residuals = Y - aprod(X, fgls_beta)
    if robust == false || typeof(robust) == BasicVariance
        # s2 = sum(residuals.^2) / (length(Y) - length(ols_beta))
        # variance = s2 .* inv(X' * X)
        variance = BasicVariance(X, residuals, length(fgls_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return fast ? (fgls_beta, variance) : linearfitresult(FGLS, X, Y, fgls_beta, residuals, variance, robust) 
end

# #########   Iterated FGLS   #########

function iteratedfgls_linfit{T<:Number}(X::Array{T}, Y::Array{T}, iterations::Int=1, intercept=true, robust=false; fast=false)
    X = intercept ? addintercept(X) : nointercept(X)
    # (1) Execute OLS regression to compute residuals:
    itfgls_beta = (X' * X) \ (X' * Y)
    residuals = Y - aprod(X, itfgls_beta)
    # (2) Use residuals to estimate Ω iteratively:
    iter::Int = 0
    for it = 1:iterations
        # println("it[$it] ")
        iter += 1 
        omega = diagm(squeeze(residuals, 2).^2)
        try
            itfgls_beta = (X' * inv(omega) * X) \ (X' * inv(omega) * Y)
            # println("\nitgls_beta[$itfgls_beta]")
            residuals = Y - aprod(X, itfgls_beta)
        catch 
            break
        end
    end
    if robust == false || typeof(robust) == BasicVariance
        # s2 = sum(residuals.^2) / (length(Y) - length(ols_beta))
        # variance = s2 .* inv(X' * X)
        variance = BasicVariance(X, residuals, length(itfgls_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return fast ? (itfgls_beta, variance) : linearfitresult(IteratedFGLS, X, Y, itfgls_beta, residuals, variance, robust, iter) 
end