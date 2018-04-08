# -----------------------------------------------
# Non-Linear Regression routines
# -----------------------------------------------

# #########   NON-LINEAR REGRESSION   #########

function nlinfit{T<:Number}(X::Array{T}, Y::Array{T}, regtype=Optimize())
    if typeof(regtype) == Optimize
        return optimize_nlinfit(X, Y, regtype.method, regtype.modelfun, regtype.beta0, regtype.intercept, regtype.robust)
    end
end

# #########   OPTIMIZE   #########

function optimize_nlinfit{T<:Number}(X::Array{T}, Y::Array{T}, method, modelfun, beta0, intercept=true, robust=false)
    # modelfun, beta0, method::AbsOptim=NewtonTrustRegion()
    # test__modelfun(modelfun, X[1:2,:], beta0)

    X = intercept ? addintercept(X) : nointercept(X)
    # Residual function given the model:
    residual(_beta_) = sum((Y - modelfun(X, _beta_)).^2)    
    optimresult = optimize(residual, beta0, method)   
    optimize_beta = Optim.minimizer(optimresult)
    residuals = Y - modelfun(X, optimize_beta)
    # mse = Optim.minimum(optimresult)
    if robust == false || typeof(robust) == BasicVariance
        # s2 = sum(residuals.^2) / (length(Y) - length(ols_beta))
        # variance = s2 .* inv(X' * X)
        variance = BasicVariance(X, residuals, length(optimize_beta))
    elseif typeof(robust) == HCEVariance
        variance = HCEVariance(X, residuals)
    end
    return nlinfitresult(Optimize, X, Y, optimize_beta, residuals, variance, robust, Optim.converged(optimresult), Optim.iterations(optimresult))    
end

function test__modelfun(modelfun, x, b)
    try
        modelfun(x, b)
        return true
    catch
        error("modelfun is not correctly defined")
    end
end
        