# Linear Regression Tests

# OLS test:
OLSRegression = OLS(intercept=true, robust=false)
res = linregress(X, Y, OLSRegression)

# OLS robust test:
OLSRegression = OLS(intercept=true, robust=HCEVariance())
res = solve(X, Y, OLSRegression)

# FGLS test
FGLSRegression = FGLS(intercept=true, robust=HCEVariance())
res = linregress(X, Y, FGLSRegression)

# IteratedFGLS test
IteratedFGLSRegression = IteratedFGLS(10, intercept=true, robust=HCEVariance())
res = linregress(X, Y, IteratedFGLSRegression)

true