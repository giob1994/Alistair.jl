using Alistair
using Base.Test

# TEST OPTIONS
const TEST_SIZE_1 = 10
const PRINT_RESULTS = false
RESULTS = Dict()

beta = [0.7; 0.45]
X = hcat(ones(TEST_SIZE_1,1), collect(1.0:TEST_SIZE_1))
Y = X * beta + 0.1*randn(TEST_SIZE_1, 1)
# Y = [1.0 2.0 1.3 3.75 2.25]'

if false
@__test issingletonarray(X) == false
@__test isrowarray(X) == false
@__test iscolumnarray(X) == true
@__test hasintercept(X) == false

@__test aprod(X, 1.0) == X
@__test aprod(1, X) == X
@__test aprod(X, [1]) == X
@__test aprod([1], X) == X
@__test aprod(X', Y) == X' * Y
end

if true
# Test for all types
RESULTS["TIME_OLS_INIT"] = @elapsed OLSRegression = OLS(intercept=true, robust=false)

# OLS test:
RESULTS["TIME_OLS_SOLVE "] = @elapsed res = linregress(X, Y, OLSRegression)
#@__test squeeze(res.beta,2) == beta
#println("OLSRegression")
PRINT_RESULTS ? println(res) : 0

OLSRegression = OLS(intercept=true, robust=HCEVariance())
RESULTS["TIME_OLS_SOLVE_HCE"] = @elapsed res = solve(X, Y, OLSRegression)
#@__test res.beta == [0.785; 0.425]
#println("OLSRegression")
PRINT_RESULTS ? println(res) : 0

# FGLS test
RESULTS["TIME_FGLS_INIT"] = @elapsed FGLSRegression = FGLS(intercept=true, robust=HCEVariance())
RESULTS["TIME_FGLS_SOLVE"] = @elapsed res = linregress(X, Y, FGLSRegression)
#println("FGLSRegression")
PRINT_RESULTS ? println(res) : 0

# IteratedFGLS test
RESULTS["TIME_IFGLS_INIT"] = @elapsed IteratedFGLSRegression = IteratedFGLS(10, intercept=true, robust=HCEVariance())
RESULTS["TIME_IFGLS_SOLVE"] = @elapsed res = linregress(X, Y, IteratedFGLSRegression)
#println("IteratedFGLS")
PRINT_RESULTS ? println(res) : 0
end

# Non-Linear test
# Optimize test

f(X_, beta_) = exp.(beta_[1]*X_[:,1] + beta_[2]*log.(X_[:,2]))

Ynlin = f(X, beta) + 0.1*randn(TEST_SIZE_1, 1)

RESULTS["TIME_OPTIM_INIT"] = @elapsed OptimizeRegression = Optimize(f, [1.0, 1.0])
RESULTS["TIME_OPTIM_SOLVE"] = @elapsed res = nlinfit(X, Ynlin, OptimizeRegression)
PRINT_RESULTS ? println(res) : 0

# Print timing results:
println("#-- TIMING RESULTS --#")
present(RESULTS)