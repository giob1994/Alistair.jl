module Alistair

    using Optim

    export
        # Types
        LinearRegressionType,
        OLS,
        GLS,
        FGLS,
        IteratedFGLS,
        GLMRegressionType,
        Logit,
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
        # GLM Regressions
        glmregress,
        logit_glmfit,
        # Non-Linear Regressions
        nlinfit,
        optimize_nlinfit,
        # DistribTools
        PDF,
        logLike
    
    include("regtools.jl")
    include("distribtools.jl")
    include("types.jl")
    include("linregress.jl")
    include("glmregress.jl")
    include("nlinregress.jl")
    include("genericregress.jl")

end # module end