# -----------------------------------------------
# Linear Regression routines
# -----------------------------------------------

"""
Linear regression function
"""

function linregress(X::Array{T}, Y::Array{T}, regtype=OLS(); fast=false) where T<:Number
    if isdefined(regtype, :intercept)
        X = regtype.intercept ? addintercept(X) : nointercept(X)
    end 
    fitted_beta, iter = regtype(X, Y)
    residuals = Y - aprod(X, fitted_beta)
    if regtype.robust == false || typeof(regtype.robust) == BasicVariance
        variance = BasicVariance(X, residuals, length(fitted_beta))
    elseif typeof(regtype.robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    if fast 
        return (fitted_beta, variance)
    elseif iter == 0
        # The regression algorithm is not iterative:
        return linearfitresult(regtype, X, Y, fitted_beta, residuals, variance, regtype.robust) 
    else
        # The regression algorithm is iterative:
        return linearfitresult(regtype, X, Y, fitted_beta, residuals, variance, regtype.robust, iter)
    end
end

function linregress(X::Array{T1}, Y::Array{T2}, regtype=OLS(); fast=false) where {T1<:Number, T2<:Number}
    X, Y = promote(X, Y)
    return linregress(X, Y, regtype; fast=fast)
end

# #########   OLS   #########

function (olsreg::OLS)(X::Array{T}, Y::Array{T}) where {T<:Number}
    # Simple OLS form:
    return (X' * X) \ (X' * Y), 0
end

# #########   GLS   #########

function (glsreg::GLS)(X::Array{T}, Y::Array{T}, omega::Array{T}) where {T<:Number}
    # Use the supplied Ω:
    return (X' * inv(omega) * X) \ (X' * inv(omega) * Y), 0
end

# #########   FGLS   #########

function (fglsreg::FGLS)(X::Array{T}, Y::Array{T}) where {T<:Number}
    # (1) Execute OLS regression to compute residuals:
    residuals = Y - X * ((X' * X) \ (X' * Y))
    # (2) Use residuals to estimate Ω:
    omega = diagm(squeeze(residuals, 2).^2)
    return (X' * inv(omega) * X) \ (X' * inv(omega) * Y), 0 
end

# #########   Iterated FGLS   #########

function (iteratedfglsreg::IteratedFGLS)(X::Array{T}, Y::Array{T}) where {T<:Number}
    itfgls_beta = (X' * X) \ (X' * Y)
    residuals = Y - aprod(X, itfgls_beta)
    # (2) Use residuals to estimate Ω iteratively:
    iter::Int = 0
    for it = 1:iteratedfglsreg.iterations
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
    return itfgls_beta, iter
end