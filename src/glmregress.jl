# -----------------------------------------------
# Generalized Linear Models routines
# -----------------------------------------------

function glmregress{T<:Number}(X::Array{T}, Y::Array{T}, regtype=false; fast=false)
    if regtype == false
        error("glmregress() - No regression type specified")
    elseif regtype == Logit()
        if iscategorical(Y)
            logit_glmfit(X, Y, regtype.intercept, regtype.robust, fast=fast)
        else
            error("Y is not categorical")
        end
    end
end

# #########   LOGIT   #########

function logit_glmfit{T<:Number}(X::Array{T}, Y::Array{T}, intercept=false, robust=false; fast=false)
    X = intercept ? addintercept(X) : nointercept(X)
    # Residual function given the model:
    theta0 = ones(size(X)[2])
    objective(theta) = -logLike(X, Y, theta, Logistic())
    optimresult = optimize(objective, theta0, NewtonTrustRegion()) 
    optimize_theta= Optim.minimizer(optimresult)
    residuals = Y - logistic(X * optimize_theta)
    if robust == false || typeof(robust) == BasicVariance
        variance = BasicVariance(X, residuals, length(optimize_theta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return glmfitresult(Optimize, X, Y, optimize_theta, residuals, variance, robust, Optim.converged(optimresult), Optim.iterations(optimresult))
end
