using Alistair

# PERFORMANCE COMPARISON OPTIONS
const TEST_SIZE_1 = 1000

Profile.clear()

#=
println("---- LINEAR REGRESSION ----")

beta = [0.7; 0.45]
X = hcat(ones(TEST_SIZE_1,1), collect(1.0:TEST_SIZE_1))
Y = X * beta + 0.1*randn(TEST_SIZE_1, 1)

olsreg = OLS()

olsreg(X, Y)
@time res = olsreg(X, Y)

linregress(X, Y, olsreg)
@time res = linregress(X, Y, olsreg)

linregress(X, Y, OLS(intercept=true, robust=HCEVariance()))
@time res = linregress(X, Y, OLS(intercept=true, robust=HCEVariance()))

linregress(X, Y, FGLS(intercept=true, robust=HCEVariance()))
@time res = linregress(X, Y, FGLS(intercept=true, robust=HCEVariance()))

linregress(X, Y, IteratedFGLS(10, intercept=true, robust=HCEVariance()))
@time res = linregress(X, Y, IteratedFGLS(10, intercept=true, robust=HCEVariance()))


println("---- NONLINEAR REGRESSION ----")

f(X_, beta_) = exp.(beta_[1]*X_[:,1] + beta_[2]*log.(X_[:,2]))

Ynlin = f(X, beta) + 0.1*randn(TEST_SIZE_1, 1)

OptimizeRegression = Optimize(f, [1.0, 1.0])

res = nlinfit(X, Ynlin, OptimizeRegression)
@time res = nlinfit(X, Ynlin, OptimizeRegression)
=#

println("---- GENERAL LINEAR MODEL REGRESSION ----")

beta_cat = [0.5; 2.0]
X_cat = hcat(ones(TEST_SIZE_1,1), collect(linspace(-2.2,2.0,TEST_SIZE_1)))
Y_cat = round.(Alistair.logistic(X_cat * beta_cat))

res_logit = glmregress(X_cat, Y_cat, Logit())
@time res_logit = glmregress(X_cat, Y_cat, Logit())
println(res_logit)