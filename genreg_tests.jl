# Non-Linear Regression Tests

f(X_, beta_) = exp.(beta_[1]*X_[:,1] + beta_[2]*log.(X_[:,2]))
Ynlin = f(X, beta) + 0.1*randn(TEST_SIZE_1, 1)

# Optimize test
OptimizeRegression = Optimize(f, [1.0, 1.0])
res = nlinfit(X, Ynlin, OptimizeRegression)

true