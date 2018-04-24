#using Alistair

# GLM Regression tests

# TEST OPTIONS
const TEST_SIZE_GLM = 10
#const PLOT = true

beta_cat = [0.5; 2.0]
X_cat = hcat(ones(TEST_SIZE_GLM,1), collect(linspace(-2.2,2.0,TEST_SIZE_GLM)))
Y_cat = round.(Alistair.logistic(X_cat * beta_cat))

res_logit = glmregress(X_cat, Y_cat, Logit())
#println(res_logit.residuals)

true