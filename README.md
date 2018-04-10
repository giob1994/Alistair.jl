<h1 align="center">
  <br>
  <a href="https://github.com/giob1994/Alistair.jl"><img src="./assets/logo-compressor.png" alt="Alistair.jl" width="100"></a>
  <br>
  <a>Alistair.jl</a>
  <br>
</h1>

<h3 align="center">A <i>minimal</i> regression library for <a href="http://julialang.org" target="_blank">Julia</a></h3>
<p align="center">
 <img src="https://img.shields.io/badge/Version-beta%200.2-a380bc.svg">
   &nbsp;
 <a href="http://julialang.org">
 	<img src="https://img.shields.io/badge/Julia-v0.6-brightgreen.svg">
 </a>
 &nbsp;
 <a href="https://travis-ci.org/giob1994/Alistair.jl">
 	<img src="https://travis-ci.org/giob1994/Alistair.jl.svg?branch=master">
 </a>
 <br>
 <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
 &nbsp;
 <a href="https://opensource.org/licenses/MIT">
 	<img src="https://img.shields.io/badge/license-MIT-blue.svg">
 </a>
 <br>
</p>

<p align="center">
 <img src="./assets/example.png" width="800">
</p>

## Install

To install **Alistair.jl** simply run:

```julia
Pkg.clone("https://github.com/giob1994/Alistair.jl.git")
```
#### Dependencies
- **[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)** – non-linear regression and optimization backend.
- **[Rmath.jl](https://github.com/JuliaStats/Rmath.jl)** – distributions backend.

## Features

- [x] **Linear Regressions**: OLS, GLS, Feasible GLS, Iterated FGLS.
- [x] **Robust Variance Matrices**: simple and HCE (Eicker-White) variance.
- [x] **Non-Linear Regression**: generic solver for *any* `Y = f(X, β)` model.
- [x] **T-test & Confidence Intervals** for estimated coefficients.
- [ ] *Logit/Probit, Multivariate Regressions*.
- [ ] *Syntactic Function Interpreter* for non-linear models.
- [ ] *Compatibility with DataFrames*.

#### 


## Why Alistair.jl?

While Julia already has some packages for regression, like **[GLM.jl](https://github.com/JuliaStats/GLM.jl)** or **[Regression.jl](https://github.com/lindahua/Regression.jl)**, sometimes a simple, bare-bones regression package is *just what is needed*. For example, *GLM.jl* accepts **only** *DataFrames* as input type, and the syntax relies on high-level constructs. 

**Alistair.jl**, on the other hand, tries to be ***minimal***, ***intuitive*** and ***fast***.

#### A Simple Example

Take the folllowing OLS (Ordinary Least Squares) regression:

```julia
beta = [0.7; 0.45]
X = hcat(ones(100,1), collect(1.0:100))
Y = X * beta + 0.1*randn(100, 1)
```

Using **Alistair.jl** we solve this with:

```julia
using Alistair

result = linregress(X, Y, OLS(intercept=true, robust=true))
```

Compare it to the equivalent *GLM.jl* code:

```julia
using Compat, DataFrames, StatsBase, GLM

data = DataFrame(X1=X[:,2], Y1=Y[:])
OLS_result = glm(@formula(Y1 ~ X1), data, Normal(), IdentityLink())
```

and the *Regression.jl* code:

```julia
ret = solve(linearreg(X, Y; bias=1.0);
            solver=GD(),
            reg=SqrL2Reg(1.0e-3),
            options=Options(verbosity=:iter, grtol=1.0e-6 * n))
```

## Basic Usage

**Alistair.jl** works at different levels of abstraction, so that if you *want*, it is possible to access the main solution routines directly for even (marginally) faster computation. 

*However*, because low-level equals risky, the best idea is to *go at your own speed*: if you are unsure of what function does what, or you are not 100% confident about the data you are handling, the higher-level functions will be happy to try and check issues for you. Don't drink *and* regress!

#### Preliminaries

Alistair.jl is *minimal*, that is, it already assumes that you are aware of the data you are using... at least a bit!

These is is a cheatsheet of good-to-know assumptions Alistair.jl makes:

- `Y [response array]` 
	- is a 1D array containing only numbers: `Float64/32` type is recommended.
	- Julia **does** make a distinction between `A::Array{Float64,1}` and `B::Array{Float64,2}` even though `B` may consist of a single column of data *identical* to that in `A`: Alistair.jl tries to work around this and generally should execute fine. If you have problem with "left-over" singleton dimensions, consider using [`squeeze()`](https://docs.julialang.org/en/stable/stdlib/arrays/#Base.squeeze).
- `X [regressor array]` 
	- is 1D/2D array containing only numbers: `Float64/32` type is recommended.
	- usually `X` is supposed to contain *more than one regressor variable*: Alistair.jl assumes that each **ROW is an OBSERVATION** and each **COLUMN is a VARIABLE**. `solve()` tries to be robust toward incorrectly specified input data, so it will check that `X` and `Y` have "compatible lengths" (i.e. `Y` has as many elements as `X` has rows) and may even try to transpose the arrays. **General advice**: *be careful and know your data before calling any Alistair.jl solver function*!

#### Using `solve(X, Y, regtype)`

The most simple command the Alistair.jl exposes is `solve()`, which takes care of routing your regression call to the right solver - *whatever type of regression you have decided to tun* -, and also checks up that the data is consistent and well-defined.

*Example*:

```julia
# Do a simple OLS regression of Y over regressors in X:

res = solve(X, Y, OLS())

# NOTE:	OLS() defaults to: OLS(intercept=true, robust=false)
```

#### Using `linregress(X, Y, regtype)`

If the `regtype` argument in `solve()` indicates a *linear* regression, then after some checks the function `linregress()` is called. If you want to access this function directly, you can do so:

*Example*:

```julia
# Do a simple OLS regression of Y over regressors in X:
# NOTE: this is equivalent to the previous example with
#       solve()

res = linregress(X, Y, OLS())
```

Alistair.jl has specific functions to solve each type of regression, so this call to `linregress()` is routed to:

```julia
ols_linfit(X, Y, intercept=true, robust=false)
```

which you can call yourself too, if you so want! Alistair.jl exports `ols_linfit()` for the brave and intrepid.

#### Robust variance matrices

As you might have notices, `OLS()` has an argument called `robust`. Alistair.jl allows to specify the way of computing the variance-covariance matrix of the regression, and specifically allows for the **Eicker-White HEC** form.

*Example*:
```julia
# Do a simple OLS regression of Y over regressors in X,
# but use the "Heteroskedastic Robust" form for the
# variance-covariance matrix.

res = linregress(X, Y, OLS(robust=HCEVariance()))
```

#### Printing results

We all like pretty printing when it comes to results! Alistair.jl defines for every result type a *nice* overload of Julia `Base.show()` method, so a nice output can be showed without hassle by using `print()`, `println()` or `show()`!

*Example*:
```julia
julia> println(linregress(X, Y, OLS()))

# ------------- RegResult [ Alistair.OLS ] ------------- #

 N of obs.: 	      10
 MSE:       	0.093079

-------------------------------------------------------------------------------------------
     Y     |      β         σ [S.E.]       t          P > |t|      [95% Conf. Interval] 
-------------------------------------------------------------------------------------------
    X[1]   |   0.776401     0.073686     10.537        0.000       0.629030   0.408779
    X[2]   |   0.432530     0.011876     36.422        0.000       0.923773   0.456281
-------------------------------------------------------------------------------------------

 Robustness: false

# ------------- EndResult [ Alistair.OLS ] ------------- #
```

## Types

Alistair specifies its own types for regressions and more. This is a useful cheetsheet:

### Regression Types

#### Abstract Types

| Alistair.jl Type         |       Supertype        |
|--------------------------|------------------------|
| `AbstractRegressionType` | None |
| `LinearRegressionType`   | `AbstractRegressionType` |
| `NonLinearRegressionType`| `AbstractRegressionType` |

#### Linear Regression Types

| Regression   | Alistair.jl Type        | Default form |
|--------------|-------------------------|---------------|
| OLS | `OLS` | `OLS(intercept=true, robust=false)` |
| GLS | `GLS` | `GLS(omega; intercept=true, robust=false)` <br> where `omega` is the mastrix chose for the GLS "sandwich" estimator |
| FGLS | `FGLS` | `FGLS(intercept=true, robust=false)` |
| Iterated FGLS | `IteratedFLGS` | `IteratedFGLS(iterations; intercept=true, robust=false)` <br> where `iterations` is the *maximum* number of iterations allowed: `IteratedFLGS` may stop earlier because convergence has been reached |

#### Non-Linear Regression Types

| Regression   | Alistair.jl Type        | Default form |
|--------------|-------------------------|---------------|
| Numerical optimization-based non-linear regression| `Optimize` | ` Optimize(modelfun, beta0, intercept=true, robust=false; method=NewtonTrustRegion())` |

### Result Types
All types in this category inherit from the abstract `AbstractFittingResult` type. These types are used for output when the regression is compelted:

| Regression Type  | Alistair.jl Result Type |   Fields |
|------------------|-------------------------|----------|
| Any | `genericfitresult` | `callertype` <br> `beta` <br> `residuals` <br> `mse` <br> `variance` |
| Linear | `linearfitresult` |`callertype` <br> `beta` <br> `residuals` <br> `mse` <br> `variance` <br> `robust` <br> `iterations`  |
|NonLinear | `nlinfitresult` |`callertype` <br> `beta` <br> `residuals` <br> `mse` <br> `variance` <br> `robust` <br> `converged` <br> `iterations` |

### Estimation Types

These are "support" types, that allow for an easier and more user-friendly specification of some tasks during the estimation phase:

#### Variance Matrix Types

All types in this category inherit from the abstract `AbstractVarianceMatrix` type. Note that all Variance Matrix Types **are functions too**! You can call them as 
```julia
myVariance = <VarianceMatrixTypeYouNeed>(...)
```

| Method           | Alistair.jl  Type | Function form |
|------------------|-------------------|---------------|
| Default | `BasicVariance` | `BasicVariance(X, residuals, k)` <br> where `k` are the degree of freedom (number of parametes in `beta`) |
| HCE (Eicker-White) | `HCEVariance` | `HCEVariance(X, residuals)` |


## Note on Performance

**Alistair.jl** is coded in standard Julia language operators like `\\`, `\*` and `inv()`. Since Julia automatically produces code that uses LAPACK functions [[1]](https://docs.julialang.org/en/stable/stdlib/linalg/), there is no real use in fiddling with LAPACK functions *inside* Julia. However, it might interesting to explore usage of SIMD functions and the parallel programming capabilities of Julia to accelerate regressions with datasets in the order of 10⁵-10⁶ observations or more.