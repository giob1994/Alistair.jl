# -----------------------------------------------
# Non-Linear Regression routines
# -----------------------------------------------

# #########   NON-LINEAR REGRESSION   #########

function nlinfit{T<:Number}(X::Array{T}, Y::Array{T}, regtype=Optimize(); fast=false)
    if isdefined(regtype, :intercept)
        X = regtype.intercept ? addintercept(X) : nointercept(X)
    end
    fitted_beta, converged, iter = regtype(X, Y)
    residuals = Y - regtype.modelfun(X, fitted_beta)
    # mse = Optim.minimum(optimresult)
    if regtype.robust == false || typeof(regtype.robust) == BasicVariance
        variance = BasicVariance(X, residuals, length(fitted_beta))
    elseif typeof(regtype.robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    if fast 
        return (fitted_beta, variance)
    else
        return nlinfitresult(regtype, X, Y, fitted_beta, residuals, variance, regtype.robust, converged, iter)
    end
end

# #########   OPTIMIZE   #########

function (optimize_nlinfit::Optimize)(X::Array{T}, Y::Array{T}) where {T<:Number}
    # modelfun, beta0, method::AbsOptim=NewtonTrustRegion()
    # test__modelfun(modelfun, X[1:2,:], beta0)

    # Residual function given the model:
    residual(_beta_) = sum((Y - optimize_nlinfit.modelfun(X, _beta_)).^2)    
    optimresult = optimize(residual, optimize_nlinfit.beta0, optimize_nlinfit.method)   
    return Optim.minimizer(optimresult), Optim.converged(optimresult), Optim.iterations(optimresult)
end

function test__modelfun(modelfun, x, b)
    try
        modelfun(x, b)
        return true
    catch
        error("modelfun is not correctly defined")
    end
end
        