using Compat, DataFrames, StatsBase
using Compat.Test
using GLM
using Alistair

# PERFORMANCE COMPARISON OPTIONS
const TEST_SIZE_1 = 100000

Profile.clear()

beta = [0.7; 0.45]
X = hcat(ones(TEST_SIZE_1,1), collect(1.0:TEST_SIZE_1))
Y = X * beta + 0.1*randn(TEST_SIZE_1, 1)

@time data = DataFrame(X1=X[:,2], Y1=Y[:])
@time OLS_result = glm(@formula(Y1 ~ X1), data, Normal(), IdentityLink())

@time OLSRegression = OLS(intercept=true, robust=false)
@time result = linregress(X, Y, OLSRegression)
