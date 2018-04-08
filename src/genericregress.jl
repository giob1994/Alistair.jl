# -----------------------------------------------
# Generic Regression and Solver routines
# -----------------------------------------------

# using Optim
# include("types.jl")

# #########   GENERIC SOLVER   #########

function solve{T<:Number}(X::Array{T}, Y::Array{T}, regtype::AbstractRegressionType)
    # Test for compatibility and correctedness of input:
    if issingletonarray(X) 
        error("X is a singleton Array - Misspecified input data")
    elseif issingletonarray(Y)
        error("Y is a singleton Array - Misspecified input data")
    elseif !(size(X)[1] == size(Y)[1] || size(X)[2] == size(Y)[2])
        error("X and Y seem not to have matching dimension - Misspecified input data")
    #elseif size(X)[2] == size(Y)[2]
    #    X = X'
    #    Y = Y'
    end
    # Dispatch to correct regression function:
    if issubtype(typeof(regtype), LinearRegressionType)
        linregress(X, Y, regtype)
    elseif issubtype(typeof(regtype), NonLinearRegressionType)
        nlinfit(X, Y, regtype)
    end
end