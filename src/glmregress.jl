# -----------------------------------------------
# Generalized Linear Models routines
# -----------------------------------------------

# using Optim
# include("distribtools.jl")
# include("types.jl")

function glmregress{T<:Number}(X::Array{T}, Y::Array{T}, regtype; fast=false)
    if !(typeof(regtype)<:GLMRegressionType)
        error("glmregress() - Incorrect GLMRegressionType specified")
    else
        if isdefined(regtype, :intercept)
            X = regtype.intercept ? addintercept(X) : nointercept(X)
        end
        fitted_beta, residuals, converged, iter = regtype(X, Y)
        if regtype.robust == false || typeof(regtype.robust) == BasicVariance
            variance = BasicVariance(X, residuals, length(fitted_beta))
        elseif typeof(regtype.robust) == HCEVariance
            variance = HCEVariance(X, residuals)
        end
        if fast 
            return (fitted_beta, variance)
        else
            return glmfitresult(regtype, X, Y, fitted_beta, residuals, variance, regtype.robust, converged, iter)
        end
    end
end

# #########   LOGIT   #########

function (logit_glmfit::Logit)(X::Array{T}, Y::Array{T}) where {T<:Number}
    theta0 = ones(size(X)[2])
    objective(theta) = -logLike(X, Y, theta, Logistic) + 0.05*norm(theta)
    optimresult = optimize(objective, theta0, NewtonTrustRegion())
    fitted_beta = Optim.minimizer(optimresult)
    residuals = Y - Alistair.logistic(aprod(X, fitted_beta))
    return fitted_beta, residuals, Optim.converged(optimresult), Optim.iterations(optimresult)
end
