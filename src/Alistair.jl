module Alistair

    using Optim

    export
        # Types
        LinearRegressionType,
        OLS,
        GLS,
        FGLS,
        IteratedFGLS,
        NonLinearRegressionType,
        Optimize,
        genericfitresult,
        linearfitresult,
        nlinfitresult,
        AbstractVarianceMatrix,
        BasicVariance,
        HCEVariance,
        # Solver
        solve,
        # Linear Regressions
        linregress,
        ols_linfit,
        gls_linfit,
        fgls_linfit,
        iteratedfgls_linfit,
        # Non-Linear Regressions
        nlinfit,
        optimize_nlinfit
    
    include("types.jl")
    include("linregress.jl")
    include("genericregress.jl")
    include("regtools.jl")

end # module end