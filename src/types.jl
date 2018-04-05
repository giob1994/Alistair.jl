# -----------------------------------------------
# Types
# -----------------------------------------------

# using Optim

import Base.show

# Import types from Optim.jl
const AbsOptim = Optim.AbstractOptimizer 
const AbsRes   = Optim.OptimizationResults


# #########   REGRESSION TYPES   #########

"""
## Regression Types

Every regression has a specific type inherited by the abstract type `AbstractRegressionType`.
"""
abstract type AbstractRegressionType end

"""
## Linear Regression

The abstract `LinearRegressionType` is the supertype for all *linear* regressions.
"""
abstract type LinearRegressionType<:AbstractRegressionType end

"""
### OLS (Ordinary Least Squares) Regression

Type that defines the simple OLS regression.
"""
struct OLS<:LinearRegressionType
    intercept::Bool
    robust::Any

    function OLS(; intercept::Bool=true, robust=false)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(intercept, robust)
    end
end

"""
### GLS (Generalized Least Squares) Regression

Type that defines the GLS estimator. Since this is a theoretical estimator, the matrix Ω must be provided.
"""
struct GLS<:LinearRegressionType
    omega
    intercept::Bool
    robust::Any

    function GLS{T<:Number}(omega::Array{T}; intercept::Bool=true, robust=false)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(omega, intercept, robust)
    end
end 

"""
### FGLS (Feasible Generalized Least Squares) Regression

Type that defines the FGLS estimator: Ω is estimated by means of the residuals from an OLS regression.
"""
struct FGLS<:LinearRegressionType
    intercept::Bool
    robust::Any

    function FGLS(; intercept::Bool=true, robust=false)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(intercept, robust)
    end
end 

""" 
### Iterated FGLS Regression

Type that defines the *Iterated* FGLS estimator: in this case, the process of estimation for Ω through residuals is iterated until convergence.
"""
struct IteratedFGLS<:LinearRegressionType
    iterations::Int
    intercept::Bool
    robust::Any

    function IteratedFGLS(iterations::Int=1; intercept::Bool=true, robust=false)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        if iterations <= 0
            error("Number of iterations must be => 1")
        end
        new(iterations, intercept, robust)
    end
end 

"""
## Non-Linear Regression

The abstract `NonLinearRegressionType` is the supertype for all *non-linear* regressions.
"""
abstract type NonLinearRegressionType<:AbstractRegressionType end

"""
### 'Optimize' Regression

Type that defines a regression where the relationship between response and regressors is a function. In this case, the package *Optim.jl* is used to carry out the minimization of the sum of squared residuals.
"""
struct Optimize<:NonLinearRegressionType
    method::AbsOptim
    modelfun
    beta0
    intercept::Bool
    robust::Any

    function Optimize{T<:Number}(modelfun, beta0::Array{T}, intercept::Bool=true, robust=false; method::AbsOptim=NewtonTrustRegion())
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(method, modelfun, beta0, intercept, robust)
    end
end


# #########   FITTING/RESULT TYPES   #########

"""
## Fitting Types

Every regression produces a collection of results/estimates.
Every output is thus packaged in a precise type, depending on the originating regression type, and inherits from the abstract type `AbstractFittingResult`.
"""
abstract type AbstractFittingResult end

function Base.show(io::IO, fitresult::AbstractFittingResult)
    # println(io, "£$(money.pounds).$(money.shillings)s.$(money.pence)d")
    println("\n- RegResult [ $(fitresult.callertype) ] -\n")
    println(" Beta: $(fitresult.beta)\n")
    #present(fitresult.beta)
    println(" MSE: $(fitresult.mse)\n")
    print(" Variance: ")
    present(fitresult.variance)
    try
        robust = fitresult.robust
        println("\n\n Robustness: $(robust)")
    catch end
    try
        converged = fitresult.converged
        println("\n Converged: $(converged)")
    catch end
    try
        iterations = fitresult.iterations
        if iterations > 0 println("\n [ Iterations: $(iterations) ]") end
    catch end
    println("\n- EndResult [ $(fitresult.callertype) ] -")
end

"""
### Generic Fitting Result

Type that implements a neutral fitting result. 
Should not be used in actual implementations.
"""
struct genericfitresult<:AbstractFittingResult
    callertype
    beta
    residuals
    mse
    variance

    function genericfitresult{T<:Number}(callertype::Any, beta::Array{T}, residuals::Array{T}, mse::T, variance::Array{T})
        new(callertype, beta, residuals, mse, variance)
    end
end

"""
### Linear Fitting Result

Type that wraps results for outputs of a `LinearRegressionType` regression.
"""
struct linearfitresult<:AbstractFittingResult
    callertype
    beta
    residuals
    mse
    variance
    robust::Any
    iterations::Int

    function linearfitresult{T<:Number}(callertype::Any, X::Array{T}, Y::Array{T}, beta::Array{T}, residuals::Array{T}, variance::Array{T}, robust::Any, iterations::Int=0)
        mse = sum(residuals.^2)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(callertype, beta, residuals, mse, variance, robust, iterations)
    end
end

"""
### Non-Linear Fitting Result

Type that wraps results for outputs of a `NonLinearRegressionType` regression.
"""
struct nlinfitresult<:AbstractFittingResult
    callertype
    beta
    residuals
    variance
    mse
    robust::Any
    converged::Bool
    iterations::Int

    function nlinfitresult{T<:Number}(callertype::Any, X::Array{T}, Y::Array{T}, beta::Array{T}, residuals::Array{T}, variance::Array{T}, robust::Any, converged::Bool, iterations::Int=0)
        """
        beta = Optim.minimizer(optimresult)
        residuals = (Y - modelfun(X, beta))
        mse = Optim.minimum(optimresult)
        converged = Optim.converged(optimresult)
        """
        mse = sum(residuals.^2)
        if robust != false && !(typeof(robust) <: AbstractVarianceMatrix)
            error("Robust option is unrecognized")
        end
        new(callertype, beta, residuals, variance, mse, robust, converged, iterations)
    end
end


# #########   ESTIMATION TYPES   #########

"""
## Estimation Types

Additional types that help with the estimation process of the regressions.

### Variance Matrices

Variance-covariance matrices are key elements in regression analysis, and require care.
Different types of *standard* and *robust* variance matrices are provided, and can be used standalone, if required.
All variance matrices types inherit from the abstract type `AbstractVarianceMatrix`.
"""
abstract type AbstractVarianceMatrix end

"""
### Basic-Naive Variance Matrix

Type that implements the naive variance-covariance matrix.
"""
struct BasicVariance<:AbstractVarianceMatrix

    function BasicVariance()
        new()
    end

    function BasicVariance{T<:Number}(X::Array{T}, residuals::Array{T}, k::Number)
        s2 = sum(residuals.^2) / (length(residuals) - k)
        return s2 .* inv(X' * X)
    end
end

"""
### HCE (Eicker-White) Heteroskedastic Consistent Variance Matrix

Type that implements the HCE variance-covariance matrix à la Eicker-White.
"""
struct HCEVariance<:AbstractVarianceMatrix

    function HCEVariance()
        new()
    end

    function HCEVariance{T<:Number}(X::Array{T}, residuals::Array{T})
        return inv(X' * X) * (X' * diagm(squeeze(residuals, 2).^2) * X) * inv(X' * X)
    end
end