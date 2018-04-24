###!/usr/bin/env julia
 
#Start Test Script
using Alistair
using Base.Test

# TEST OPTIONS
const TEST_SIZE_1 = 10
const PRINT_RESULTS = false

# GENERATE TEST DATA
beta = [0.7; 0.45]
X = hcat(ones(TEST_SIZE_1,1), collect(1.0:TEST_SIZE_1))
Y = X * beta + 0.1*randn(TEST_SIZE_1, 1)

# Run test
tic()

@time @test Alistair.issingletonarray(X) == false
@time @test Alistair.isrowarray(X) == false
@time @test Alistair.iscolumnarray(Y) == true
@time @test Alistair.hasintercept(X) == true

@time @test Alistair.aprod(X, 1.0) == X
@time @test Alistair.aprod(1, X) == X
@time @test Alistair.aprod(X, [1]) == X
@time @test Alistair.aprod([1], X) == X
@time @test Alistair.aprod(X', Y) == X' * Y

@time @test include("linreg_tests.jl")
@time @test include("genreg_tests.jl")
@time @test include("glmreg_tests.jl")

# End test
toc()