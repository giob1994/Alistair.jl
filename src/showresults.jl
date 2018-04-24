# -----------------------------------------------
# Generalized Linear Models routines
# -----------------------------------------------

function Base.show(io::IO, fitresult::AbstractFittingResult)
    # println(io, "£$(money.pounds).$(money.shillings)s.$(money.pence)d")
    f_ = "-------------"
    #s_ = "  "
    println("\n# $f_ RegResult [ $(typeof(fitresult.callertype)) ] $f_ #\n")
    #println(" Beta: $(fitresult.beta)\n")
    #present(fitresult.beta)
    println(" N of obs.: \t$(@sprintf("%8d", fitresult.N))")
    println(" MSE:       \t$(@sprintf("%8f", fitresult.mse))\n")
    println("$f_$f_$f_$f_$f_$f_$f_")
    println("     Y     |      β         σ [S.E.]       t          P > |t|      [95% Conf. Interval] ")
    println("$f_$f_$f_$f_$f_$f_$f_")
    for i in 1:length(fitresult.beta)
        print("    X[$i]   |")
        print(" $(@sprintf("%10f", fitresult.beta[i]))  ")
        print(" $(@sprintf("%10f", sqrt(fitresult.variance[i,i])))  ")
        print(" $(@sprintf("%8.3f", fitresult.tstat[i]))    ")
        print(" $(@sprintf("%8.3f", fitresult.tsig[i]))      ")
        print(" $(@sprintf("%8f", fitresult.confint[i][1]))   $(@sprintf("%8f", fitresult.confint[i][2]))")
        println("")
    end
    println("$f_$f_$f_$f_$f_$f_$f_")
    #print(" Variance: ")
    #present(fitresult.variance)
    try
        robust = fitresult.robust
        println("\n Robustness: $(robust)")
    catch end
    try
        converged = fitresult.converged
        println("\n Converged: $(converged)")
    catch end
    try
        iterations = fitresult.iterations
        if iterations > 0 println("\n [ Iterations: $(iterations) ]") end
    catch end
    println("\n# $f_ EndResult [ $(typeof(fitresult.callertype)) ] $f_ #")
end

function Base.show(io::IO, fitresult::glmfitresult)
    # println(io, "£$(money.pounds).$(money.shillings)s.$(money.pence)d")
    f_ = "-------------"
    #s_ = "  "
    println("\n# $f_ RegResult [ $(typeof(fitresult.callertype)) ] $f_ #\n")
    #println(" Beta: $(fitresult.beta)\n")
    #present(fitresult.beta)
    println(" N of obs.: \t$(@sprintf("%8d", fitresult.N))")
    if typeof(fitresult.callertype) == Logit
        println(" D:        \t$(@sprintf("%8f", fitresult.fit))\n")
    else
        println(" MSE:       \t$(@sprintf("%8f", fitresult.fit))\n")
    end
    println("$f_$f_$f_$f_$f_$f_$f_")
    println("     Y     |      β         σ [S.E.]       t          P > |t|      [95% Conf. Interval] ")
    println("$f_$f_$f_$f_$f_$f_$f_")
    for i in 1:length(fitresult.beta)
        print("    X[$i]   |")
        print(" $(@sprintf("%10f", fitresult.beta[i]))  ")
        print(" $(@sprintf("%10f", sqrt(fitresult.variance[i,i])))  ")
        print(" $(@sprintf("%8.3f", fitresult.tstat[i]))    ")
        print(" $(@sprintf("%8.3f", fitresult.tsig[i]))      ")
        print(" $(@sprintf("%8f", fitresult.confint[i][1]))   $(@sprintf("%8f", fitresult.confint[i][2]))")
        println("")
    end
    println("$f_$f_$f_$f_$f_$f_$f_")
    #print(" Variance: ")
    #present(fitresult.variance)
    try
        robust = fitresult.robust
        println("\n Robustness: $(robust)")
    catch end
    try
        converged = fitresult.converged
        println("\n Converged: $(converged)")
    catch end
    try
        iterations = fitresult.iterations
        if iterations > 0 println("\n [ Iterations: $(iterations) ]") end
    catch end
    println("\n# $f_ EndResult [ $(typeof(fitresult.callertype)) ] $f_ #")
end