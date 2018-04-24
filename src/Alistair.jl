module Alistair

    using Optim     # non-linear solver backend
    using Rmath     # distributions backend

    export
        # Types
        LinearRegressionType,
        GLMRegressionType,
        NonLinearRegressionType,
        # Solver
        solve,
        # Linear Regressions
        linregress,
        OLS,
        GLS,
        FGLS,
        IteratedFGLS,
        # GLM Regressions
        glmregress,
        Logit,
        # Non-Linear Regressions
        nlinfit,
        Optimize,
        # Variance matrix
        AbstractVarianceMatrix,
        BasicVariance,
        HCEVariance,
        # Results
        genericfitresult,
        linearfitresult,
        nlinfitresult,
        # DistribTools
        PDF,
        CDF,
        logLike,
        Normal,
        Logistic,
        Student,
        # Tools
        __test,
        present
    
    include("regtools.jl")
    include("distribtools.jl")
    include("types.jl")
    include("linregress.jl")
    include("glmregress.jl")
    include("nlinregress.jl")
    include("genericregress.jl")
    include("showresults.jl")

end # module end