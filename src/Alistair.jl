module Alistair

    using Optim

    export
        solve
        linregress
        ols_linfit
        nlinfit
    
    include("types.jl")
    include("linregress.jl")
    include("genericregress.jl")
    include("mixfunctions.jl")

end # module end