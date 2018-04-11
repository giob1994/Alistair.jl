using Alistair

# PERFORMANCE COMPARISON OPTIONS
const TEST_SIZE_1 = 1000

Profile.clear()

beta = [0.7; 0.45]
X = hcat(ones(TEST_SIZE_1,1), collect(1.0:TEST_SIZE_1))
Y = X * beta + 0.1*randn(TEST_SIZE_1, 1)

olsreg = OLS()

olsreg(X, Y)
@time res = olsreg(X, Y)

linregress(X, Y, olsreg)
@time res = linregress(X, Y, olsreg)

#=
@time res = linregress(X, Y, olsreg)
@time res = linregress(X, Y, olsreg)
=#
